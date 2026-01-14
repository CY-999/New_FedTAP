# src/defenses/fedtap.py
# -*- coding: utf-8 -*-
"""
FedTAP: Trust-aware Temporal Aggregation for Robust FL
Strictly aligned with FedTAP proposal (Eq.(3)~(14)).

This implementation fits the current repo interface:
aggregate(updates: List[Tensor], global_model: Tensor, client_ids: List[int], bn_mask: Tensor, global_model_obj: nn.Module, ...) -> (agg_update, stats)

Key components:
  3.1 Observer-based prediction (AR(K) + forgetting factor + ridge) -> predicts g_hat_{t+1}
      - Eq.(3)(4) via sufficient-statistics recursion (equivalent to online ridge with forgetting)
  Temporal residual per client:
      - Eq.(5)(6), computed per layer then normalized and summed
  3.2 Multi-scale detection:
      - Spatial cue: PCA subspace of last W aggregated updates (Eq.(7))
      - Robust normalization by Median/MAD (Eq.(8))
      - z-score fusion (Eq.(9))
      - Logistic credibility (Eq.(10)), with kappa from MAD(z)
      - tau update (Eq.(12))
      - Trust propagation (Eq.(11)) + hysteresis (low-trust slow recovery)
  3.3 Trust-aware adaptive aggregation:
      - Weight mapping (Eq.(13))
      - Plug-in base aggregator (Eq.(14))
      - Isolation rule: trust below theta_min for L consecutive rounds -> isolate for cooldown rounds
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .baselines import BaseDefense


# -----------------------------
# Helpers: robust statistics
# -----------------------------
def _median(x: torch.Tensor) -> torch.Tensor:
    # torch.median on 1D is fine
    return x.median()

def _mad(x: torch.Tensor, med: torch.Tensor) -> torch.Tensor:
    return (x - med).abs().median()

def _robust_z(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, float, float]:
    """
    Return robust z-score: (x - median) / (1.4826 * MAD + eps)
    Also return median and MAD as python floats.
    """
    med = _median(x)
    mad = _mad(x, med)
    scale = 1.4826 * mad + eps
    z = (x - med) / scale
    return z, float(med.item()), float(mad.item())


# -----------------------------
# Layer slicing & sampling
# -----------------------------
@dataclass
class _LayerSpec:
    name: str
    start: int
    end: int
    abs_idx: torch.Tensor  # 1D Long indices into full flattened vector (already BN-filtered and sampled)
    dim_eff: int


# -----------------------------
# Predictor state per layer (AR(K) with forgetting factor)
# -----------------------------
@dataclass
class _PredState:
    # Rolling buffer of sampled g vectors for this layer (oldest -> newest)
    buf: deque  # holds torch.Tensor vectors (float32), maxlen = K+1

    # Sufficient statistics for online ridge with forgetting:
    # S = sum (X^T X), t = sum (X^T y)   where X=[g_{t-1},...,g_{t-K}], y=g_t
    S: torch.Tensor  # (K,K) float64
    t: torch.Tensor  # (K,) float64


# -----------------------------
# PCA state per layer (subspace from last W aggregated updates)
# -----------------------------
@dataclass
class _PCAState:
    buf: deque  # holds last W aggregated updates (sampled) for this layer, each float32
    basis: Optional[torch.Tensor]  # (dim_eff, d_s) float32
    d_s: int
    var_ratio: float


class FedTAP(BaseDefense):
    def __init__(self, config: Dict[str, Any] = None, base_defense: Optional[BaseDefense] = None):
        super().__init__(config)
        cfg = config or {}

        # ---- Proposal defaults / hyperparams ----
        # Predictor (Eq.(3)(4))
        self.K = int(cfg.get("K", 3))                       # AR order, typical 2~4
        self.gamma_pred = float(cfg.get("gamma_pred", 0.95))# forgetting factor
        self.rho = float(cfg.get("rho", 1e-4))              # ridge term
        self.W_pred = int(cfg.get("W_pred", 20))            # rolling buffer size hint (kept for compatibility)

        # PCA subspace (Eq.(7))
        self.W_pca = int(cfg.get("W_pca", 20))              # typical 10~30
        self.var_retained = float(cfg.get("var_retained", 0.90))  # 0.90~0.95
        self.subspace_dim_max = int(cfg.get("subspace_dim_max", 16))

        # Fusion (Eq.(9))
        self.alpha = float(cfg.get("alpha", 1.0))
        self.beta = float(cfg.get("beta", 1.0))

        # Logistic credibility (Eq.(10)(12))
        self.gamma_tau = float(cfg.get("gamma_tau", 0.90))  # 0.8~0.95
        self.c_tau = float(cfg.get("c_tau", 3.0))           # 2~4
        self.tau = None  # adaptive threshold tau_t

        # Trust propagation (Eq.(11)) + hysteresis
        self.beta_theta = float(cfg.get("beta_theta", 0.90))    # 0.8~0.99
        self.hysteresis_th = float(cfg.get("hysteresis_th", 0.30))
        self.hysteresis_up_factor = float(cfg.get("hysteresis_up_factor", 0.25))  # slow recovery when low trust

        # Isolation rule
        self.theta_min = float(cfg.get("theta_min", 0.30))
        self.isolation_L = int(cfg.get("isolation_L", 3))
        self.cooldown_rounds = int(cfg.get("cooldown_rounds", 3))

        # Weight mapping (Eq.(13))
        self.lam = float(cfg.get("lambda", 5.0))  # typical 4~6

        # Candidate global step (Eq.(5))
        self.eta = float(cfg.get("eta", 1.0))

        # Efficiency knob: effective dims per layer (d_eff in proposal)
        # If None -> use all non-BN params of each layer. Otherwise sample at most this many per layer.
        self.max_features_per_layer = cfg.get("max_features_per_layer", None)
        if self.max_features_per_layer is not None:
            self.max_features_per_layer = int(self.max_features_per_layer)

        self.eps = float(cfg.get("eps", 1e-12))

        # ---- States ----
        self._layers: Optional[List[_LayerSpec]] = None
        self._pred: Dict[str, _PredState] = {}
        self._pca: Dict[str, _PCAState] = {}

        # Per-client trust / streak / cooldown
        self.theta = defaultdict(lambda: 1.0)
        self.low_streak = defaultdict(int)
        self.cooldown = defaultdict(int)

        # For early-round prediction fallback
        self._prev_agg_update: Optional[torch.Tensor] = None

        # Plug-in base aggregator (Eq.(14))
        # If None, we do weighted FedAvg (normalized) as base.
        self.base_defense = base_defense

    # -----------------------------
    # Internal: build layer index mapping
    # -----------------------------
    @torch.no_grad()
    def _ensure_layers(
        self,
        global_model: torch.Tensor,
        bn_mask: Optional[torch.Tensor],
        global_model_obj: Optional[Any],
    ):
        if self._layers is not None:
            return

        d = int(global_model.numel())
        device = global_model.device

        # If no model object, treat whole vector as one "layer"
        if global_model_obj is None or (not hasattr(global_model_obj, "named_parameters")):
            # BN mask handling
            if bn_mask is not None:
                active = (~bn_mask).nonzero(as_tuple=False).view(-1)
            else:
                active = torch.arange(d, device=device, dtype=torch.long)

            abs_idx = active
            if self.max_features_per_layer is not None and abs_idx.numel() > self.max_features_per_layer:
                # deterministic-ish sampling
                g = torch.Generator(device=device)
                g.manual_seed(0)
                perm = torch.randperm(abs_idx.numel(), generator=g, device=device)[: self.max_features_per_layer]
                abs_idx = abs_idx.index_select(0, perm)

            self._layers = [_LayerSpec(name="all", start=0, end=d, abs_idx=abs_idx, dim_eff=int(abs_idx.numel()))]
            return

        # Build slices by iterating parameters in a stable order
        layers: List[_LayerSpec] = []
        offset = 0
        for name, p in global_model_obj.named_parameters():
            n = int(p.numel())
            start, end = offset, offset + n
            offset = end

            # active indices inside this slice (exclude BN if bn_mask is provided)
            if bn_mask is not None:
                # bn_mask is over full vector
                active = (~bn_mask[start:end]).nonzero(as_tuple=False).view(-1) + start
            else:
                active = torch.arange(start, end, device=device, dtype=torch.long)

            if active.numel() == 0:
                continue

            abs_idx = active
            if self.max_features_per_layer is not None and abs_idx.numel() > self.max_features_per_layer:
                g = torch.Generator(device=device)
                # seed tied to layer name for reproducibility
                g.manual_seed(abs(hash(name)) % (2**31))
                perm = torch.randperm(abs_idx.numel(), generator=g, device=device)[: self.max_features_per_layer]
                abs_idx = abs_idx.index_select(0, perm)

            layers.append(_LayerSpec(name=name, start=start, end=end, abs_idx=abs_idx, dim_eff=int(abs_idx.numel())))

        # Safety: if mapping failed, fallback to whole vector
        if not layers:
            if bn_mask is not None:
                active = (~bn_mask).nonzero(as_tuple=False).view(-1)
            else:
                active = torch.arange(d, device=device, dtype=torch.long)
            layers = [_LayerSpec(name="all", start=0, end=d, abs_idx=active, dim_eff=int(active.numel()))]

        self._layers = layers

    @torch.no_grad()
    def _take(self, vec: torch.Tensor, layer: _LayerSpec) -> torch.Tensor:
        # vec: (D,)
        return vec.index_select(0, layer.abs_idx).float()

    # -----------------------------
    # 3.1 Prediction: update stats + predict g_hat_{t+1} per layer
    # -----------------------------
    @torch.no_grad()
    def _pred_update_and_predict(self, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return dict: layer_name -> g_hat_{t+1} (sampled vector for that layer)
        """
        assert self._layers is not None
        device = g.device

        pred_out: Dict[str, torch.Tensor] = {}

        for layer in self._layers:
            x_t = self._take(g, layer)  # sampled g_t for this layer

            if layer.name not in self._pred:
                # init state
                buf = deque(maxlen=self.K + 1)
                buf.append(x_t)
                S = torch.zeros((self.K, self.K), device=device, dtype=torch.float64)
                t = torch.zeros((self.K,), device=device, dtype=torch.float64)
                self._pred[layer.name] = _PredState(buf=buf, S=S, t=t)

                # early fallback prediction
                if self._prev_agg_update is not None:
                    pred_out[layer.name] = x_t + self._take(self._prev_agg_update, layer)
                else:
                    pred_out[layer.name] = x_t.clone()
                continue

            st = self._pred[layer.name]
            st.buf.append(x_t)

            # Update sufficient statistics when we have enough history:
            # y = g_t, X = [g_{t-1}, ..., g_{t-K}]
            if len(st.buf) >= self.K + 1:
                y = st.buf[-1]  # newest
                X = [st.buf[-2 - k] for k in range(self.K)]  # K vectors

                # forgetting
                st.S.mul_(self.gamma_pred)
                st.t.mul_(self.gamma_pred)

                # Accumulate X^T X and X^T y via inner products
                for a in range(self.K):
                    xa = X[a]
                    st.t[a] += torch.dot(xa, y).double()
                    for b in range(a, self.K):
                        xb = X[b]
                        v = torch.dot(xa, xb).double()
                        st.S[a, b] += v
                        if b != a:
                            st.S[b, a] += v

            # Solve for coefficients (ridge): (S + rho I) w = t
            # If S is near-singular in early rounds, ridge keeps it stable.
            S_reg = st.S + (self.rho * torch.eye(self.K, device=device, dtype=torch.float64))
            try:
                w = torch.linalg.solve(S_reg, st.t)  # (K,) float64
            except RuntimeError:
                w = torch.zeros((self.K,), device=device, dtype=torch.float64)
                w[0] = 1.0

            # Predict g_hat_{t+1} = sum_{k=0}^{K-1} w[k] * g_{t-k}
            # Use feature list: [g_t, g_{t-1}, ..., g_{t-K+1}]
            feats = [st.buf[-1 - k] for k in range(min(self.K, len(st.buf)))]
            # pad if not enough feats
            while len(feats) < self.K:
                feats.append(feats[-1])

            g_hat = torch.zeros_like(feats[0])
            for k in range(self.K):
                g_hat.add_(feats[k], alpha=float(w[k].item()))

            pred_out[layer.name] = g_hat.float()

        return pred_out

    # -----------------------------
    # 3.2 Subspace: PCA from last W aggregated updates (per layer)
    # -----------------------------
    @torch.no_grad()
    def _pca_basis_from_buffer(self, X: torch.Tensor) -> Tuple[Optional[torch.Tensor], int, float]:
        """
        X: (n, d_eff) aggregated updates from recent rounds (n<=W_pca)
        Return (basis U: (d_eff, d_s), d_s, var_ratio)
        """
        n, d = X.shape
        if n < 2:
            return None, 0, 0.0

        # center
        Xc = X - X.mean(dim=0, keepdim=True)

        # Gram matrix (n x n), eigen-decomp
        G = Xc @ Xc.t()  # (n,n)
        # eigenvalues ascending -> flip
        evals, evecs = torch.linalg.eigh(G)
        evals = torch.clamp(evals, min=0.0)
        order = torch.argsort(evals, descending=True)
        evals = evals.index_select(0, order)
        evecs = evecs.index_select(1, order)

        total = float(evals.sum().item()) + self.eps
        if total <= self.eps:
            return None, 0, 0.0

        # choose d_s to retain var_retained, cap by subspace_dim_max and rank
        cumsum = torch.cumsum(evals, dim=0)
        ratio = (cumsum / total)
        d_s = int(torch.searchsorted(ratio, torch.tensor(self.var_retained, device=ratio.device)) + 1)
        d_s = max(1, min(d_s, self.subspace_dim_max, n))

        # Build basis U = Xc^T V / sqrt(evals)
        V = evecs[:, :d_s]  # (n, d_s)
        svals = torch.sqrt(evals[:d_s] + self.eps)  # (d_s,)
        U = (Xc.t() @ V) / svals.unsqueeze(0)       # (d, d_s)

        # Orthonormalize U (QR)
        U, _ = torch.linalg.qr(U, mode="reduced")

        var_ratio = float(cumsum[d_s - 1].item() / total)
        return U.float(), d_s, var_ratio

    @torch.no_grad()
    def _get_pca_basis(self, layer_name: str, device: torch.device) -> Optional[torch.Tensor]:
        st = self._pca.get(layer_name, None)
        if st is None or st.basis is None:
            return None
        return st.basis.to(device=device)

    # -----------------------------
    # Base aggregation (Eq.(14))
    # -----------------------------
    @torch.no_grad()
    def _weighted_fedavg(self, updates: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """
        Stable weighted average: sum w_i * Δ_i / sum w_i
        """
        s = float(weights.sum().item())
        if s <= self.eps:
            return torch.mean(torch.stack(updates), dim=0)
        out = torch.zeros_like(updates[0])
        for u, w in zip(updates, weights):
            out.add_(u, alpha=float(w.item()))
        out.div_(s + self.eps)
        return out

    # -----------------------------
    # Main API
    # -----------------------------
    @torch.no_grad()
    def aggregate(
        self,
        updates: List[torch.Tensor],
        global_model: torch.Tensor,
        client_ids: Optional[List[int]] = None,
        bn_mask: Optional[torch.Tensor] = None,
        global_model_obj: Optional[Any] = None,
        round_num: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {"method": "FedTAP", "num_clients": 0}

        m = len(updates)
        device = updates[0].device
        dtype = updates[0].dtype

        if client_ids is None:
            client_ids = list(range(m))

        g = global_model.to(device=device, dtype=dtype)

        # Build layer specs once
        self._ensure_layers(g, bn_mask, global_model_obj)

        # 3.1 Prediction -> g_hat_{t+1} per layer (sampled vectors)
        g_hat_map = self._pred_update_and_predict(g)

        # 3.2 Spatial cue -> PCA basis from last W aggregated updates (do NOT include current round)
        # Prepare basis for each layer
        pca_basis_map: Dict[str, Optional[torch.Tensor]] = {}
        pca_var_map: Dict[str, float] = {}
        if self._layers is not None:
            for layer in self._layers:
                st = self._pca.get(layer.name, None)
                if st is not None and st.basis is not None:
                    pca_basis_map[layer.name] = st.basis.to(device=device)
                    pca_var_map[layer.name] = st.var_ratio
                else:
                    pca_basis_map[layer.name] = None
                    pca_var_map[layer.name] = 0.0

        # Compute r_{t,i} and s_{t,i} for each client (Eq.(6)(7))
        r = torch.zeros((m,), device=device, dtype=torch.float32)
        s = torch.zeros((m,), device=device, dtype=torch.float32)

        assert self._layers is not None
        for j, u in enumerate(updates):
            # per-layer normalized residual sum
            r_sum = 0.0
            s_sum = 0.0

            for layer in self._layers:
                u_l = self._take(u, layer)                    # Δ_{t,i} sampled
                g_l = self._take(g, layer)                    # g_t sampled
                g_hat_l = g_hat_map[layer.name]               # ĝ_{t+1} sampled

                # candidate global if only client i applied (Eq.(5))
                g_i = g_l + (self.eta * u_l)

                # temporal residual (Eq.(6)), per-layer normalized before summation
                denom = float(torch.norm(g_hat_l).item()) + self.eps
                r_layer = float(torch.norm(g_i - g_hat_l).item()) / denom
                r_sum += r_layer

                # spatial deviation (Eq.(7)): ||Δ - P_S(Δ)||
                U = pca_basis_map.get(layer.name, None)
                if U is None:
                    # no basis yet
                    s_layer = 0.0
                else:
                    # projection
                    y = U.t().matmul(u_l)        # (d_s,)
                    proj = U.matmul(y)           # (dim_eff,)
                    s_layer = float(torch.norm(u_l - proj).item())
                s_sum += s_layer

            r[j] = float(r_sum)
            s[j] = float(s_sum)

        # Robust normalization (Eq.(8))
        r_tilde, r_med, r_mad = _robust_z(r, self.eps)
        s_tilde, s_med, s_mad = _robust_z(s, self.eps)

        # Anomaly score (Eq.(9))
        z = (self.alpha * r_tilde) + (self.beta * s_tilde)

        # κ_t from MAD(z) (Eq.(10))
        z_med = _median(z)
        z_mad = _mad(z, z_med)
        kappa = float((1.4826 * z_mad + self.eps).item())

        # Update τ_t (Eq.(12))
        tau_target = float((z_med + (self.c_tau * z_mad)).item())
        if self.tau is None:
            self.tau = tau_target
        else:
            self.tau = (self.gamma_tau * float(self.tau)) + ((1.0 - self.gamma_tau) * tau_target)

        tau = float(self.tau)

        # Logistic credibility c_{t,i} (Eq.(10)):
        # c = 1 / (1 + exp((z - tau)/kappa)) = sigmoid(-(z - tau)/kappa)
        c = torch.sigmoid(-(z - tau) / (kappa + self.eps)).float()

        # Trust propagation (Eq.(11)) + hysteresis + isolation
        theta_list = []
        w = torch.zeros((m,), device=device, dtype=torch.float32)
        isolated = []

        for j, cid in enumerate(client_ids):
            th_old = float(self.theta[cid])

            # hysteresis: when low trust, recovery is slower
            c_j = float(c[j].item())
            if th_old < self.hysteresis_th and c_j > th_old:
                c_eff = th_old + self.hysteresis_up_factor * (c_j - th_old)
            else:
                c_eff = c_j

            th_new = (self.beta_theta * th_old) + ((1.0 - self.beta_theta) * c_eff)
            th_new = max(0.0, min(1.0, th_new))
            self.theta[cid] = th_new
            theta_list.append(th_new)

            # update streak / cooldown
            if th_new < self.theta_min:
                self.low_streak[cid] += 1
            else:
                self.low_streak[cid] = 0

            if self.cooldown[cid] > 0:
                self.cooldown[cid] -= 1
                w[j] = 0.0
                isolated.append(int(cid))
                continue

            # Isolation if below theta_min for L consecutive rounds
            if self.low_streak[cid] >= self.isolation_L:
                self.cooldown[cid] = self.cooldown_rounds
                w[j] = 0.0
                isolated.append(int(cid))
                continue

            # Weight mapping (Eq.(13))
            w[j] = float(torch.exp(torch.tensor(-self.lam * (1.0 - th_new), device=device)).item())

        # Apply weights into base aggregator (Eq.(14))
        # We implement the reweighting layer; base aggregator is plug-and-play:
        #   g_{t+1} = F_base(g_t, {w_i * Δ_i})
        weighted_updates = [u.mul(float(wi.item())) for u, wi in zip(updates, w)]

        if self.base_defense is None:
            # Stable base: weighted FedAvg (normalized)
            agg_update = self._weighted_fedavg(updates, w)
            base_stats = {"base": "WeightedFedAvg(normalized)"}
        else:
            # Pass reweighted updates to the chosen base defense (plug-and-play)
            # Keep only args that the base defense supports
            import inspect
            sig = inspect.signature(self.base_defense.aggregate)
            params = sig.parameters
            base_kwargs = {"updates": weighted_updates, "global_model": g}
            if "bn_mask" in params:
                base_kwargs["bn_mask"] = bn_mask
            if "root_gradient" in params and "root_gradient" in kwargs:
                base_kwargs["root_gradient"] = kwargs["root_gradient"]
            if "device" in params and "device" in kwargs:
                base_kwargs["device"] = kwargs["device"]
            if "global_model_obj" in params:
                base_kwargs["global_model_obj"] = global_model_obj
            if "val_loader" in params and "val_loader" in kwargs:
                base_kwargs["val_loader"] = kwargs["val_loader"]

            agg_update, base_stats = self.base_defense.aggregate(**base_kwargs)
            if base_stats is None:
                base_stats = {"base": type(self.base_defense).__name__}

        # Update PCA buffers using current aggregated update (adds \barΔ_t into recent W set)
        # This matches: { \barΔ_u }_{u=t-W+1}^t used for subspace S (proposal Sec 3.2).
        for layer in self._layers:
            delta_bar = self._take(agg_update, layer).to(device=device)
            if layer.name not in self._pca:
                self._pca[layer.name] = _PCAState(
                    buf=deque(maxlen=self.W_pca),
                    basis=None,
                    d_s=0,
                    var_ratio=0.0
                )
            self._pca[layer.name].buf.append(delta_bar)

        # Recompute PCA basis for next round (online refresh)
        for layer in self._layers:
            st = self._pca[layer.name]
            if len(st.buf) >= 2:
                X = torch.stack(list(st.buf), dim=0)  # (n, dim_eff)
                U, d_s, var_ratio = self._pca_basis_from_buffer(X)
                st.basis = U
                st.d_s = d_s
                st.var_ratio = var_ratio

        # Save prev agg update for early-round prediction fallback
        self._prev_agg_update = agg_update.detach()

        keep_ratio = float((w > 0).float().mean().item())
        stats = {
            "method": "FedTAP",
            "round": int(round_num) if round_num is not None else None,
            "num_clients": m,
            "r_med": r_med,
            "r_mad": r_mad,
            "s_med": s_med,
            "s_mad": s_mad,
            "z_med": float(z_med.item()),
            "z_mad": float(z_mad.item()),
            "tau": float(tau),
            "kappa": float(kappa),
            "cred_mean": float(c.mean().item()),
            "theta_mean": float(sum(theta_list) / max(1, len(theta_list))),
            "keep_ratio": keep_ratio,
            "isolated": isolated,
            "weights": [float(x) for x in w.tolist()],
            "pca_var_mean": float(sum(pca_var_map.values()) / max(1, len(pca_var_map))) if pca_var_map else 0.0,
        }

        # Attach base stats
        if isinstance(base_stats, dict):
            for k, v in base_stats.items():
                stats[f"base_{k}"] = v

        return agg_update, stats
