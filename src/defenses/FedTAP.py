# src/defenses/fedtap.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .baselines import BaseDefense


def _robust_z(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Robust z-score via median/MAD:
      z = (x - med) / (1.4826 * MAD + eps)
    """
    if x.numel() == 0:
        return x
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / (1.4826 * mad + eps)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


class FedTAP(BaseDefense):
    """
    Efficient temporal robust aggregator with alignment + contribution + self-purify.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        # ---- Temporal reference direction ----
        self.beta_ref = float(cfg.get("beta_ref", 0.90))  # EMA for reference direction
        self._ref_dir: Optional[torch.Tensor] = None

        # ---- Trust propagation ----
        self.beta_theta = float(cfg.get("beta_theta", 0.90))   # EMA for per-client trust
        self.init_trust = float(cfg.get("init_trust", 1.0))
        self.hysteresis_th = float(cfg.get("hysteresis_th", 0.30))
        self.hysteresis_up_factor = float(cfg.get("hysteresis_up_factor", 0.25))

        # ---- Fusion weights (alignment + sign + contribution) ----
        self.w_cos = float(cfg.get("w_cos", 1.0))
        self.w_sign = float(cfg.get("w_sign", 1.0))
        self.w_contrib = float(cfg.get("w_contrib", 1.0))
        self.z_clip = float(cfg.get("z_clip", 6.0))
        self.temp = float(cfg.get("temp", 1.0))  # sigmoid temperature

        # ---- Important parameter subset for sign alignment ----
        self.important_ratio = float(cfg.get("important_ratio", 0.10))  # top 10%
        self.max_important = int(cfg.get("max_important", 200_000))      # hard cap for efficiency
        self.min_important = int(cfg.get("min_important", 20_000))       # ensure not too small

        # ---- Self-purify + aggregation ----
        self.gamma_w = float(cfg.get("gamma_w", 2.0))  # trust -> weight sharpening
        self.isolate_th = float(cfg.get("isolate_th", 0.05))  # extremely low trust
        self.clip_coef = float(cfg.get("clip_coef", 2.5))     # median-norm clipping

        # ---- Contribution (validation gradient) ----
        self.use_contrib = bool(cfg.get("use_contrib", True))
        self.max_val_batches = int(cfg.get("max_val_batches", 3))  # small => fast

        # internal trust table: client_id -> theta
        self._theta: Dict[int, float] = {}

        self._round = 0

    @staticmethod
    def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return v / (v.norm() + eps)

    def _get_theta(self, cid: int) -> float:
        if cid not in self._theta:
            self._theta[cid] = float(self.init_trust)
        return float(self._theta[cid])

    def _set_theta(self, cid: int, val: float):
        self._theta[cid] = float(max(0.0, min(1.0, val)))

    def _build_important_idx(self, ref_dir: torch.Tensor) -> torch.Tensor:
        """
        Choose important parameter indices based on |ref_dir|.
        We only use this subset for sign alignment to be efficient.
        """
        d = ref_dir.numel()
        k = int(max(self.min_important, min(self.max_important, int(d * self.important_ratio))))
        k = min(k, d)

        # If d is small, just use all
        if k >= d:
            return torch.arange(d, device=ref_dir.device, dtype=torch.long)

        # topk on abs(ref_dir)
        abs_ref = ref_dir.abs()
        _, idx = torch.topk(abs_ref, k=k, largest=True, sorted=False)
        return idx

    def _compute_val_grad(
        self,
        model: Any,
        val_loader: Any,
        device: torch.device,
        bn_mask: Optional[torch.Tensor],
        max_batches: int,
        eps: float = 1e-12
    ) -> Optional[torch.Tensor]:
        """
        Compute one flattened validation gradient g (average over a few batches).
        This is used to estimate marginal contribution via dot products.
        Very efficient: one backward pass per batch, only a few batches.
        """
        if model is None or val_loader is None:
            return None

        # Keep current mode, but eval is safer (no dropout etc.)
        was_training = model.training
        model.eval()

        grads_accum = None
        seen = 0

        try:
            for bidx, (x, y) in enumerate(val_loader):
                if bidx >= max_batches:
                    break

                x = x.to(device)
                y = y.to(device)

                model.zero_grad(set_to_none=True)
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()

                flat = []
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    flat.append(p.grad.detach().reshape(-1))
                if not flat:
                    return None

                g = torch.cat(flat).to(device)

                if bn_mask is not None and bn_mask.numel() == g.numel():
                    g = g.clone()
                    g[bn_mask] = 0

                if grads_accum is None:
                    grads_accum = g.to(torch.float32)
                else:
                    grads_accum += g.to(torch.float32)

                seen += 1

            if grads_accum is None or seen == 0:
                return None

            grads_accum = grads_accum / float(seen)
            # normalize for stable cosine-like contribution
            return grads_accum / (grads_accum.norm() + eps)

        except Exception:
            return None
        finally:
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()

    def aggregate(
        self,
        updates: List[torch.Tensor],
        global_model: torch.Tensor,
        client_ids: Optional[List[int]] = None,
        bn_mask: Optional[torch.Tensor] = None,
        val_loader: Optional[Any] = None,
        device: Optional[torch.device] = None,
        global_model_obj: Optional[Any] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {"method": "FedTAP", "num_clients": 0}

        self._round += 1

        dev = device or updates[0].device
        eps = 1e-12

        m = len(updates)
        if client_ids is None or len(client_ids) != m:
            # fallback: unstable across rounds, but keeps running
            client_ids = list(range(m))
            client_ids_provided = False
        else:
            client_ids_provided = True

        # ---- mask BN params if provided ----
        proc_updates: List[torch.Tensor] = []
        for u in updates:
            uu = u
            if bn_mask is not None and bn_mask.numel() == u.numel():
                uu = uu.clone()
                uu[bn_mask] = 0
            proc_updates.append(uu)

        # ---- build / update reference direction ----
        if self._ref_dir is None:
            # init from mean direction (round 1)
            init_ref = torch.mean(torch.stack(proc_updates, dim=0), dim=0)
            self._ref_dir = self._normalize(init_ref.to(torch.float32), eps=eps)
        ref_dir = self._ref_dir.to(proc_updates[0].device)

        # important subset for sign alignment
        imp_idx = self._build_important_idx(ref_dir)

        # ---- (Optional) compute validation gradient (once per round) ----
        g_val = None
        if self.use_contrib and (val_loader is not None) and (global_model_obj is not None):
            g_val = self._compute_val_grad(
                model=global_model_obj,
                val_loader=val_loader,
                device=dev,
                bn_mask=bn_mask,
                max_batches=self.max_val_batches,
                eps=eps
            )

        # ---- compute per-client metrics ----
        cos_d = torch.empty(m, device=dev, dtype=torch.float32)
        sign_d = torch.empty(m, device=dev, dtype=torch.float32)
        contrib_d = torch.empty(m, device=dev, dtype=torch.float32)

        ref_n = self._normalize(ref_dir.to(torch.float32), eps=eps)
        ref_sign = torch.sign(ref_n.index_select(0, imp_idx))
        ref_sign[ref_sign == 0] = 1  # avoid zeros

        for i, u in enumerate(proc_updates):
            uf = u.to(torch.float32)
            un = self._normalize(uf, eps=eps)

            # 1) direction distance: 1 - cosine
            c = torch.dot(un, ref_n).clamp(-1.0, 1.0)
            cos_d[i] = 1.0 - c

            # 2) sign distance on important subset
            us = torch.sign(un.index_select(0, imp_idx))
            us[us == 0] = 1
            match = (us == ref_sign).to(torch.float32).mean()
            sign_d[i] = 1.0 - match

            # 3) contribution distance: smaller contribution => more suspicious
            if g_val is not None and g_val.numel() == uf.numel():
                # contrib = - <g_val, u> / (||u||) since g_val already normalized
                cn = -torch.dot(g_val.to(torch.float32), uf) / (uf.norm() + eps)
                # higher cn is better; turn into "badness"
                contrib_d[i] = -cn
            else:
                contrib_d[i] = 0.0

        # ---- robust normalization and fusion ----
        z_cos = _robust_z(cos_d, eps=eps)
        z_sign = _robust_z(sign_d, eps=eps)
        z_contrib = _robust_z(contrib_d, eps=eps) if (g_val is not None) else torch.zeros_like(z_cos)

        z = self.w_cos * z_cos + self.w_sign * z_sign + self.w_contrib * z_contrib
        z = torch.clamp(z, -self.z_clip, self.z_clip)

        # credibility in (0,1), higher is better
        c_round = _sigmoid((-z) / max(self.temp, 1e-6))

        # ---- update temporal trust (theta) ----
        theta_list = []
        isolated = []
        for i, cid in enumerate(client_ids):
            theta_old = self._get_theta(cid)
            theta_new = self.beta_theta * theta_old + (1.0 - self.beta_theta) * float(c_round[i].item())

            # hysteresis: slow recovery when already low
            if theta_old < self.hysteresis_th and theta_new > theta_old:
                theta_new = theta_old + self.hysteresis_up_factor * (theta_new - theta_old)

            self._set_theta(cid, theta_new)
            theta_list.append(theta_new)
            isolated.append(theta_new < self.isolate_th)

        theta_t = torch.tensor(theta_list, device=dev, dtype=torch.float32)

        # ---- self-purify updates (projection onto ref_dir for low trust) ----
        ref_denom = torch.dot(ref_n, ref_n) + eps  # ~1
        purified: List[torch.Tensor] = []

        for i, u in enumerate(proc_updates):
            uf = u.to(torch.float32)
            # projection component
            coef = torch.dot(uf, ref_n) / ref_denom
            proj = coef * ref_n

            th = float(theta_t[i].item())
            if isolated[i]:
                # extremely suspicious: keep only projection
                u_clean = proj
            else:
                # blend: low trust -> more projection
                u_clean = th * uf + (1.0 - th) * proj

            purified.append(u_clean.to(u.dtype))

        # ---- robust norm clipping (median-based) ----
        norms = torch.tensor([p.to(torch.float32).norm().item() for p in purified], device=dev, dtype=torch.float32)
        med_norm = norms.median().item()
        clip_bound = self.clip_coef * med_norm + eps

        clipped: List[torch.Tensor] = []
        for p in purified:
            pn = p.to(torch.float32).norm().item()
            if pn > clip_bound:
                scale = clip_bound / (pn + eps)
                clipped.append((p * scale).to(p.dtype))
            else:
                clipped.append(p)

        # ---- trust weights ----
        w = torch.pow(theta_t.clamp_min(0.0), self.gamma_w)
        w = w + 1e-6  # avoid all-zero
        w = w / w.sum()

        # ---- final aggregation ----
        agg = torch.zeros_like(proc_updates[0])
        for wi, ui in zip(w.tolist(), clipped):
            agg.add_(ui, alpha=float(wi))

        # ---- update reference direction EMA ----
        agg_n = self._normalize(agg.to(torch.float32), eps=eps)
        self._ref_dir = self._normalize(self.beta_ref * ref_n + (1.0 - self.beta_ref) * agg_n, eps=eps).detach()

        stats: Dict[str, Any] = {
            "method": "FedTAP",
            "round": self._round,
            "num_clients": m,
            "client_ids_provided": client_ids_provided,
            "important_k": int(imp_idx.numel()),
            "use_contrib": bool(g_val is not None),
            "cos_d": cos_d.detach().cpu().tolist(),
            "sign_d": sign_d.detach().cpu().tolist(),
            "contrib_bad": contrib_d.detach().cpu().tolist(),
            "z": z.detach().cpu().tolist(),
            "credibility": c_round.detach().cpu().tolist(),
            "theta": theta_t.detach().cpu().tolist(),
            "isolated": isolated,
            "weights": w.detach().cpu().tolist(),
            "median_norm": float(med_norm),
            "clip_bound": float(clip_bound),
            "agg_l2": float(agg.to(torch.float32).norm().item()),
        }

        return agg, stats
