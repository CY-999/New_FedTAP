# src/defenses/fedtap.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F

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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class FedTAP(BaseDefense):
    """
    Efficient temporal robust aggregator with alignment + sign + contribution + self-purify,
    enhanced by server-side convergence guard to trigger stricter filtering in late training.

    Updates in this version (per user's request):
      (1) Trust-aware aggregation uses exponential sharpening:
          w_i ∝ exp(-λ (1 - θ_i)) = exp(λ (θ_i - 1))
      (2) Credibility mapping uses logistic with τ_t and κ_t:
          c_{t,i} = σ((τ_t - z_{t,i}) / κ_t),
          where κ_t is MAD-adaptive, and τ_t is EMA-smoothed.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config or {}

        # For logging / result identification
        self.profile_name = str(cfg.get("profile_name", "FedTAP"))

        # ---- Temporal reference direction ----
        self.beta_ref = float(cfg.get("beta_ref", 0.90))
        self._ref_dir: Optional[torch.Tensor] = None

        # ---- Trust propagation ----
        self.beta_theta = float(cfg.get("beta_theta", 0.90))
        self.init_trust = float(cfg.get("init_trust", 1.0))
        self.hysteresis_th = float(cfg.get("hysteresis_th", 0.30))
        self.hysteresis_up_factor = float(cfg.get("hysteresis_up_factor", 0.25))

        # ---- Fusion weights (alignment + sign + contribution) ----
        self.w_cos = float(cfg.get("w_cos", 1.0))
        self.w_sign = float(cfg.get("w_sign", 1.0))
        self.w_contrib = float(cfg.get("w_contrib", 1.0))
        self.z_clip = float(cfg.get("z_clip", 6.0))

        # ---- Logistic credibility (τ_t, κ_t) ----
        # Backward-compat: old "temp" is treated as a scale for κ_t if "kappa_scale" not provided.
        self.kappa_scale = float(cfg.get("kappa_scale", cfg.get("temp", 1.0)))
        self.kappa_min = float(cfg.get("kappa_min", 1e-4))
        self.kappa_max = float(cfg.get("kappa_max", 10.0))

        # τ_t smoothing
        self.tau_beta = float(cfg.get("tau_beta", 0.90))          # EMA smoothing for τ_t
        self.tau_offset = float(cfg.get("tau_offset", 0.0))       # optional shift: τ_target = median(z)+offset
        tau_init = cfg.get("tau_init", None)
        self._tau_t: Optional[float] = float(tau_init) if tau_init is not None else None

        # ---- Important parameter subset for sign alignment ----
        self.important_ratio = float(cfg.get("important_ratio", 0.10))
        self.max_important = int(cfg.get("max_important", 200_000))
        self.min_important = int(cfg.get("min_important", 20_000))

        # ---- Self-purify + aggregation ----
        self.isolate_th = float(cfg.get("isolate_th", 0.05))
        self.clip_coef = float(cfg.get("clip_coef", 2.5))

        # Trust weight sharpening (exponential): w ∝ exp(-λ(1-θ))
        # Backward-compat: if "lambda_w" not provided, fall back to old "gamma_w".
        self.lambda_w = float(cfg.get("lambda_w", cfg.get("gamma_w", 2.0)))

        # ---- Contribution (validation gradient direction) ----
        self.use_contrib = bool(cfg.get("use_contrib", True))
        self.max_val_batches = int(cfg.get("max_val_batches", 3))

        # ========== Convergence Guard (val loss plateau + update norm) ==========
        self.enable_convergence_guard = bool(cfg.get("enable_convergence_guard", True))
        self.conv_warmup_rounds = int(cfg.get("conv_warmup_rounds", 30))
        self.conv_patience = int(cfg.get("conv_patience", 5))
        self.conv_ema_beta = float(cfg.get("conv_ema_beta", 0.90))
        self.conv_loss_rel_delta_th = float(cfg.get("conv_loss_rel_delta_th", 0.005))
        # threshold on agg_norm_rel_delta
        self.conv_norm_frac = float(cfg.get("conv_norm_frac", 0.55))

        # ========== Strict filtering once stable ==========
        # Logistic sharpening: strict_temp_mul now acts as κ multiplier (<1 => sharper sigmoid)
        self.strict_temp_mul = float(cfg.get("strict_temp_mul", 0.60))

        # Exponential trust-weight sharpening multiplier (compat: fall back to strict_gamma_mul)
        self.strict_lambda_mul = float(cfg.get("strict_lambda_mul", cfg.get("strict_gamma_mul", 1.70)))

        self.strict_isolate_th = float(cfg.get("strict_isolate_th", 0.12))
        self.strict_clip_mul = float(cfg.get("strict_clip_mul", 0.80))

        self.strict_z_drop = float(cfg.get("strict_z_drop", 2.2))
        self.strict_theta_floor = float(cfg.get("strict_theta_floor", 0.22))
        self.strict_cred_floor = float(cfg.get("strict_cred_floor", 0.28))
        self.strict_keep_ratio = float(cfg.get("strict_keep_ratio", 0.85))
        self.strict_hard_drop = bool(cfg.get("strict_hard_drop", False))
        self.strict_down_weight = float(cfg.get("strict_down_weight", 0.03))
        self.strict_theta_cap = float(cfg.get("strict_theta_cap", 0.70))

        # internal trust table: client_id -> theta
        self._theta: Dict[int, float] = {}

        # internal round
        self._round = 0

        # convergence guard state
        self.in_stable_phase = False
        self.stable_start_round: Optional[int] = None
        self._stable_count = 0
        self._agg_norm_ema: Optional[float] = None
        self._loss_ema: Optional[float] = None

        # cache val grad direction (optional)
        self._cached_val_grad: Optional[torch.Tensor] = None
        self._cached_val_grad_round: int = -1

        # last info for logging
        self.last_info: Dict[str, Any] = {}

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
        d = ref_dir.numel()
        k = int(max(self.min_important, min(self.max_important, int(d * self.important_ratio))))
        k = min(k, d)
        if k >= d:
            return torch.arange(d, device=ref_dir.device, dtype=torch.long)
        abs_ref = ref_dir.abs()
        _, idx = torch.topk(abs_ref, k=k, largest=True, sorted=False)
        return idx

    @torch.no_grad()
    def _compute_val_loss(
        self,
        model: Any,
        val_loader: Any,
        device: torch.device,
        max_batches: int,
    ) -> Optional[float]:
        if model is None or val_loader is None:
            return None
        was_training = model.training
        model.eval()
        total_loss = 0.0
        total_n = 0
        try:
            for bidx, batch in enumerate(val_loader):
                if bidx >= max_batches:
                    break
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    return None
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                total_loss += float(loss.item())
                total_n += int(y.numel())
            if total_n <= 0:
                return None
            return total_loss / float(total_n)
        finally:
            if was_training:
                model.train()

    def _compute_val_grad_dir(
        self,
        model: Any,
        val_loader: Any,
        device: torch.device,
        bn_mask: Optional[torch.Tensor],
        max_batches: int,
        eps: float = 1e-12
    ) -> Optional[torch.Tensor]:
        """
        One normalized validation gradient direction (flattened).
        Returns (-grad) / ||grad||, i.e., a descent direction on the validation loss.
        """
        if model is None or val_loader is None:
            return None

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
                loss = F.cross_entropy(out, y)
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
            return (-grads_accum) / (grads_accum.norm() + eps)

        except Exception:
            return None
        finally:
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()

    def _update_convergence_guard(
        self,
        round_idx: int,
        val_loss: Optional[float],
        agg_norm: Optional[float],
        eps: float = 1e-12
    ) -> Dict[str, Any]:
        """
        Convergence guard (enter strict once, then sticky).

        Signals:
        - loss_rel_delta: relative change of loss EMA
        - agg_norm_rel_delta: relative change of aggregated-update-norm EMA
        """
        info: Dict[str, Any] = {
            "candidate": False,
            "entered": False,
            "stable_start_round": self.stable_start_round,
            "val_loss": _safe_float(val_loss, default=float("nan")),
            "loss_ema": float("nan"),
            "loss_rel_delta": float("nan"),
            "agg_norm": _safe_float(agg_norm, default=float("nan")),
            "agg_norm_ema": float("nan"),
            "agg_norm_rel_delta": float("nan"),
            "agg_norm_frac": float("nan"),
        }

        if not self.enable_convergence_guard:
            return info

        beta = float(self.conv_ema_beta)

        # ---- loss EMA + relΔ ----
        if val_loss is not None and math.isfinite(float(val_loss)):
            if self._loss_ema is None:
                self._loss_ema = float(val_loss)
                loss_rel_delta = 0.0
            else:
                prev = float(self._loss_ema)
                self._loss_ema = beta * prev + (1.0 - beta) * float(val_loss)
                loss_rel_delta = abs(self._loss_ema - prev) / max(abs(prev), eps)

            info["loss_ema"] = float(self._loss_ema)
            info["loss_rel_delta"] = float(loss_rel_delta)

        # ---- agg_norm EMA + relΔ ----
        if agg_norm is not None and math.isfinite(float(agg_norm)):
            if self._agg_norm_ema is None:
                self._agg_norm_ema = float(agg_norm)
                agg_rel_delta = 0.0
            else:
                prev = float(self._agg_norm_ema)
                self._agg_norm_ema = beta * prev + (1.0 - beta) * float(agg_norm)
                agg_rel_delta = abs(self._agg_norm_ema - prev) / max(abs(prev), eps)

            info["agg_norm_ema"] = float(self._agg_norm_ema)
            info["agg_norm_rel_delta"] = float(agg_rel_delta)
            info["agg_norm_frac"] = float(agg_rel_delta)

        # ---- warmup: only track EMAs ----
        if round_idx <= int(self.conv_warmup_rounds):
            return info

        # ---- sticky strict ----
        if self.in_stable_phase:
            return info

        loss_plateau_ok = (info["loss_rel_delta"] <= float(self.conv_loss_rel_delta_th))
        norm_plateau_ok = (info["agg_norm_rel_delta"] <= float(self.conv_norm_frac))
        candidate = bool(loss_plateau_ok and norm_plateau_ok)
        info["candidate"] = candidate

        if candidate:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if (self._stable_count >= int(self.conv_patience)) and (not self.in_stable_phase):
            self.in_stable_phase = True
            self.stable_start_round = int(round_idx)
            info["entered"] = True
            info["stable_start_round"] = self.stable_start_round

        return info

    @staticmethod
    def _mad(x: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (median, MAD) for a 1D tensor x.
        MAD = median(|x - median(x)|)
        """
        med = x.median()
        mad = (x - med).abs().median()
        return med, mad + eps

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
        round_idx = self._round

        dev = device or updates[0].device
        eps = 1e-12

        m = len(updates)
        if client_ids is None or len(client_ids) != m:
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
            init_ref = torch.mean(torch.stack(proc_updates, dim=0), dim=0)
            self._ref_dir = self._normalize(init_ref.to(torch.float32), eps=eps)
        ref_dir = self._ref_dir.to(proc_updates[0].device)

        # important subset for sign alignment
        imp_idx = self._build_important_idx(ref_dir)

        # ---- server-side convergence signals ----
        val_loss = None
        if self.enable_convergence_guard and (val_loader is not None) and (global_model_obj is not None):
            val_loss = self._compute_val_loss(global_model_obj, val_loader, dev, max_batches=self.max_val_batches)

        # IMPORTANT: strict used for THIS round is determined by previous rounds only
        strict = bool(self.in_stable_phase)

        # strict effective params (other logic unchanged)
        isolate_th_eff = max(self.isolate_th, self.strict_isolate_th) if strict else self.isolate_th
        clip_coef_eff = self.clip_coef * (self.strict_clip_mul if strict else 1.0)

        # ---- (Optional) compute validation gradient direction (once per round, cached) ----
        g_val = None
        if self.use_contrib and (val_loader is not None) and (global_model_obj is not None):
            if self._cached_val_grad is None or self._cached_val_grad_round != round_idx:
                self._cached_val_grad = self._compute_val_grad_dir(
                    model=global_model_obj,
                    val_loader=val_loader,
                    device=dev,
                    bn_mask=bn_mask,
                    max_batches=self.max_val_batches,
                    eps=eps
                )
                self._cached_val_grad_round = round_idx
            g_val = self._cached_val_grad

        # ---- compute per-client metrics ----
        cos_d = torch.empty(m, device=dev, dtype=torch.float32)
        sign_d = torch.empty(m, device=dev, dtype=torch.float32)
        contrib_d = torch.empty(m, device=dev, dtype=torch.float32)

        ref_n = self._normalize(ref_dir.to(torch.float32), eps=eps)
        ref_sign = torch.sign(ref_n.index_select(0, imp_idx))
        ref_sign[ref_sign == 0] = 1

        for i, u in enumerate(proc_updates):
            uf = u.to(torch.float32)
            un = self._normalize(uf, eps=eps)

            c = torch.dot(un, ref_n).clamp(-1.0, 1.0)
            cos_d[i] = 1.0 - c

            us = torch.sign(un.index_select(0, imp_idx))
            us[us == 0] = 1
            match = (us == ref_sign).to(torch.float32).mean()
            sign_d[i] = 1.0 - match

            if g_val is not None and g_val.numel() == uf.numel():
                cos_g = torch.dot(g_val.to(torch.float32), uf) / (uf.norm() + eps)  # [-1, 1]
                contrib_d[i] = 1.0 - cos_g  # larger => more suspicious
            else:
                contrib_d[i] = 0.0

        # ---- robust normalization and fusion ----
        z_cos = _robust_z(cos_d, eps=eps)
        z_sign = _robust_z(sign_d, eps=eps)
        z_contrib = _robust_z(contrib_d, eps=eps) if (g_val is not None) else torch.zeros_like(z_cos)

        z_raw = self.w_cos * z_cos + self.w_sign * z_sign + self.w_contrib * z_contrib
        z = torch.clamp(z_raw, -self.z_clip, self.z_clip)

        # ---- NEW: logistic credibility with τ_t and κ_t (MAD-adaptive), τ_t EMA-smoothed ----
        # κ_t from MAD(z): kappa = kappa_scale * (1.4826 * MAD + eps)
        med_z, mad_z = self._mad(z, eps=eps)
        kappa_t = self.kappa_scale * (1.4826 * mad_z)
        kappa_t = torch.clamp(kappa_t, min=self.kappa_min, max=self.kappa_max)

        # τ_t target = median(z) + offset, then EMA smooth
        tau_target = float(med_z.item()) + float(self.tau_offset)
        if self._tau_t is None or (not math.isfinite(float(self._tau_t))):
            self._tau_t = float(tau_target)
        else:
            self._tau_t = float(self.tau_beta * float(self._tau_t) + (1.0 - self.tau_beta) * float(tau_target))

        # strict => sharper sigmoid by shrinking κ
        kappa_eff = float(kappa_t.item()) * (self.strict_temp_mul if strict else 1.0)
        kappa_eff = max(kappa_eff, 1e-6)

        # credibility higher is better
        c_round = _sigmoid((float(self._tau_t) - z) / kappa_eff)

        # ---- update temporal trust (theta) ----
        theta_list: List[float] = []
        for i, cid in enumerate(client_ids):
            theta_old = self._get_theta(cid)
            theta_new = self.beta_theta * theta_old + (1.0 - self.beta_theta) * float(c_round[i].item())

            # hysteresis: slow recovery for low-trust clients
            if theta_old < self.hysteresis_th and theta_new > theta_old:
                theta_new = theta_old + self.hysteresis_up_factor * (theta_new - theta_old)

            self._set_theta(cid, theta_new)
            theta_list.append(theta_new)

        theta_t = torch.tensor(theta_list, device=dev, dtype=torch.float32)

        # ---- strict suspicious rules (late-stage hardening) ----
        suspicious = torch.zeros(m, dtype=torch.bool, device=dev)
        if strict:
            suspicious = (
                (z > self.strict_z_drop)
                | (theta_t < self.strict_theta_floor)
                | (c_round < self.strict_cred_floor)
            )

            if self.strict_keep_ratio < 1.0:
                keep_k = max(1, int(math.ceil(self.strict_keep_ratio * m)))
                keep_idx = torch.topk(c_round, k=keep_k, largest=True).indices
                keep_mask = torch.zeros(m, dtype=torch.bool, device=dev)
                keep_mask[keep_idx] = True
                suspicious = suspicious | (~keep_mask)

        # isolate if extremely low trust or strict suspicious
        isolated = (theta_t < isolate_th_eff) | suspicious

        # ---- self-purify updates (projection onto ref_dir for low trust) ----
        ref_denom = torch.dot(ref_n, ref_n) + eps
        purified: List[torch.Tensor] = []

        for i, u in enumerate(proc_updates):
            uf = u.to(torch.float32)
            coef = torch.dot(uf, ref_n) / ref_denom
            proj = coef * ref_n

            th = float(theta_t[i].item())
            if strict:
                th = min(th, self.strict_theta_cap)

            if bool(isolated[i].item()):
                u_clean = proj
            else:
                u_clean = th * uf + (1.0 - th) * proj

            purified.append(u_clean.to(u.dtype))

        # ---- robust norm clipping (median-based) ----
        norms = torch.tensor([p.to(torch.float32).norm().item() for p in purified], device=dev, dtype=torch.float32)
        if strict and (~suspicious).any():
            med_norm = float(norms[~suspicious].median().item())
        else:
            med_norm = float(norms.median().item())
        clip_bound = clip_coef_eff * med_norm + eps

        clipped: List[torch.Tensor] = []
        for p in purified:
            pn = float(p.to(torch.float32).norm().item())
            if pn > clip_bound:
                scale = clip_bound / (pn + eps)
                clipped.append((p * scale).to(p.dtype))
            else:
                clipped.append(p)

        # ---- NEW: exponential trust weights (strict => larger λ) ----
        theta01 = theta_t.clamp(0.0, 1.0)
        lambda_eff = float(self.lambda_w) * (self.strict_lambda_mul if strict else 1.0)

        # w_i ∝ exp(-λ(1-θ_i)) = exp(λ(θ_i - 1))
        w = torch.exp(lambda_eff * (theta01 - 1.0)) + 1e-6

        # strict downweight / hard drop suspicious
        if strict and suspicious.any():
            if self.strict_hard_drop:
                w = torch.where(suspicious, torch.zeros_like(w), w)
            else:
                w = torch.where(suspicious, w * self.strict_down_weight, w)

        # guard against all-zero
        if float(w.sum().item()) <= 1e-12:
            if strict and suspicious.any() and (~suspicious).any():
                w = (~suspicious).to(w.dtype)
            else:
                w = torch.ones_like(w)

        w = w / (w.sum() + 1e-12)

        # ---- final aggregation ----
        agg = torch.zeros_like(proc_updates[0])
        for wi, ui in zip(w.tolist(), clipped):
            agg.add_(ui, alpha=float(wi))

        # ---- update reference direction EMA ----
        agg_n = self._normalize(agg.to(torch.float32), eps=eps)
        self._ref_dir = self._normalize(self.beta_ref * ref_n + (1.0 - self.beta_ref) * agg_n, eps=eps).detach()

        # ---- update convergence guard AFTER aggregation using agg step norm ----
        agg_l2 = float(agg.to(torch.float32).norm().item())
        conv_info: Dict[str, Any] = {}
        strict_next = strict
        entered = False
        if self.enable_convergence_guard:
            conv_info = self._update_convergence_guard(round_idx, val_loss, agg_l2, eps=eps)
            strict_next = bool(self.in_stable_phase)
            entered = bool(conv_info.get("entered", False))

        # ---- minimal decisive stats for analysis/logging ----
        suspicious_ratio = float(suspicious.to(torch.float32).mean().item())
        theta_min = float(theta_t.min().item())

        stats: Dict[str, Any] = {
            "method": "FedTAP",
            "round": int(round_idx),
            "num_clients": int(m),
            "client_ids_provided": bool(client_ids_provided),

            "profile": str(self.profile_name),
            "enable_convergence_guard": bool(self.enable_convergence_guard),

            "strict": bool(strict),
            "strict_next": bool(strict_next),
            "entered_strict": bool(entered),
            "stable_start_round": conv_info.get("stable_start_round", None),

            "val_loss": _safe_float(conv_info.get("val_loss", val_loss), default=float("nan")),
            "loss_ema": _safe_float(conv_info.get("loss_ema", None), default=float("nan")),
            "loss_rel_delta": _safe_float(conv_info.get("loss_rel_delta", None), default=float("nan")),

            "agg_norm": float(agg_l2),
            "agg_norm_frac": _safe_float(conv_info.get("agg_norm_frac", None), default=float("nan")),

            "suspicious_ratio": float(suspicious_ratio),
            "theta_min": float(theta_min),
            "clip_bound": float(clip_bound),

            # NEW: logistic & exponential-sharpening diagnostics
            "tau_t": _safe_float(self._tau_t, default=float("nan")),
            "kappa_t": float(kappa_t.item()) if isinstance(kappa_t, torch.Tensor) else _safe_float(kappa_t, default=float("nan")),
            "kappa_eff": float(kappa_eff),
            "lambda_eff": float(lambda_eff),

            # keep legacy key if your plotting expects it
            "agg_l2": float(agg_l2),
        }

        self.last_info = stats
        return agg, stats