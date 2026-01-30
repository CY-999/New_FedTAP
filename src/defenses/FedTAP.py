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
    Efficient temporal robust aggregator with alignment + contribution + self-purify,
    enhanced by server-side convergence guard (val loss plateau + aggregated-update-norm plateau)
    to trigger stricter filtering in late training (esp. against backdoors).
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
        self.max_important = int(cfg.get("max_important", 200_000))      # hard cap
        self.min_important = int(cfg.get("min_important", 20_000))       # floor

        # ---- Self-purify + aggregation ----
        self.gamma_w = float(cfg.get("gamma_w", 2.0))  # trust -> weight sharpening
        self.isolate_th = float(cfg.get("isolate_th", 0.05))  # extremely low trust
        self.clip_coef = float(cfg.get("clip_coef", 2.5))     # median-norm clipping

        # ---- Contribution (validation gradient direction) ----
        self.use_contrib = bool(cfg.get("use_contrib", True))
        self.max_val_batches = int(cfg.get("max_val_batches", 3))  # small => fast

        # ========== NEW: Convergence Guard (val loss plateau + update norm) ==========
        self.enable_convergence_guard = bool(cfg.get("enable_convergence_guard", True))
        self.conv_warmup_rounds = int(cfg.get("conv_warmup_rounds", 30))
        self.conv_patience = int(cfg.get("conv_patience", 5))
        self.conv_ema_beta = float(cfg.get("conv_ema_beta", 0.90))
        # loss plateau: relative EMA delta threshold
        self.conv_loss_rel_delta_th = float(cfg.get("conv_loss_rel_delta_th", 0.005))
        # update plateau: relative change of agg_norm EMA (|ema_t-ema_{t-1}|/|ema_{t-1}|)
        # NOTE: conv_norm_frac is a threshold on agg_norm_rel_delta (not decay-to-max fraction).
        self.conv_norm_frac = float(cfg.get("conv_norm_frac", 0.55))

        # ========== NEW: Strict filtering once stable ==========
        # sharpen credibility mapping
        self.strict_temp_mul = float(cfg.get("strict_temp_mul", 0.60))      # temp_eff = temp * mul
        # trust weight sharpening
        self.strict_gamma_mul = float(cfg.get("strict_gamma_mul", 1.70))    # gamma_eff = gamma * mul
        # stricter isolation threshold (projection-only)
        self.strict_isolate_th = float(cfg.get("strict_isolate_th", 0.12))  # isolate_th_eff = max(isolate_th, strict)
        # tighter clipping
        self.strict_clip_mul = float(cfg.get("strict_clip_mul", 0.80))      # clip_coef_eff = clip_coef * mul
        # strict suspicious rules
        self.strict_z_drop = float(cfg.get("strict_z_drop", 2.2))
        self.strict_theta_floor = float(cfg.get("strict_theta_floor", 0.22))
        self.strict_cred_floor = float(cfg.get("strict_cred_floor", 0.28))
        self.strict_keep_ratio = float(cfg.get("strict_keep_ratio", 0.85))  # keep top-k by credibility
        self.strict_hard_drop = bool(cfg.get("strict_hard_drop", False))
        self.strict_down_weight = float(cfg.get("strict_down_weight", 0.03))

        # internal trust table: client_id -> theta
        self._theta: Dict[int, float] = {}

        # internal round
        self._round = 0

        # convergence guard state
        self.in_stable_phase = False
        self.stable_start_round: Optional[int] = None
        self._stable_count = 0
        self._post_warmup_norm_inited = False
        self._agg_norm_ema: Optional[float] = None
        self._agg_norm_ema_prev: Optional[float] = None
        self._last_agg_norm: Optional[float] = None

        self._loss_ema: Optional[float] = None
        self._loss_ema_prev: Optional[float] = None

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
        (Used only when use_contrib=True.)
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
            return grads_accum / (grads_accum.norm() + eps)

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

        Use plateau (relative EMA change) instead of absolute decay:
        under multi-round attacks, update magnitudes may not monotonically shrink.
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
        }

        if not self.enable_convergence_guard:
            return info

        beta = float(self.conv_ema_beta)

        # ---- loss EMA + relΔ ----
        if val_loss is not None and math.isfinite(float(val_loss)):
            if self._loss_ema is None:
                self._loss_ema = float(val_loss)
                self._loss_ema_prev = float(val_loss)
                loss_rel_delta = 0.0
            else:
                prev = float(self._loss_ema)
                self._loss_ema_prev = prev
                self._loss_ema = beta * prev + (1.0 - beta) * float(val_loss)
                loss_rel_delta = abs(self._loss_ema - prev) / max(abs(prev), eps)

            info["loss_ema"] = float(self._loss_ema)
            info["loss_rel_delta"] = float(loss_rel_delta)

        # ---- agg_norm EMA + relΔ ----
        if agg_norm is not None and math.isfinite(float(agg_norm)):
            if self._agg_norm_ema is None:
                self._agg_norm_ema = float(agg_norm)
                self._agg_norm_ema_prev = float(agg_norm)
                agg_rel_delta = 0.0
            else:
                prev = float(self._agg_norm_ema)
                self._agg_norm_ema_prev = prev
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
        # 1) server val loss (small set)
        val_loss = None
        if self.enable_convergence_guard and (val_loader is not None) and (global_model_obj is not None):
            val_loss = self._compute_val_loss(global_model_obj, val_loader, dev, max_batches=self.max_val_batches)

        # IMPORTANT: strict used for THIS round is determined by previous rounds only
        strict = bool(self.in_stable_phase)

        # strict effective params
        temp_eff = self.temp * (self.strict_temp_mul if strict else 1.0)
        gamma_eff = self.gamma_w * (self.strict_gamma_mul if strict else 1.0)
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
                cn = -torch.dot(g_val.to(torch.float32), uf) / (uf.norm() + eps)
                contrib_d[i] = -cn
            else:
                contrib_d[i] = 0.0

        # ---- robust normalization and fusion ----
        z_cos = _robust_z(cos_d, eps=eps)
        z_sign = _robust_z(sign_d, eps=eps)
        z_contrib = _robust_z(contrib_d, eps=eps) if (g_val is not None) else torch.zeros_like(z_cos)

        z = self.w_cos * z_cos + self.w_sign * z_sign + self.w_contrib * z_contrib
        z = torch.clamp(z, -self.z_clip, self.z_clip)

        # credibility higher is better (strict => sharper)
        c_round = _sigmoid((-z) / max(temp_eff, 1e-6))

        # ---- update temporal trust (theta) ----
        theta_list = []
        for i, cid in enumerate(client_ids):
            theta_old = self._get_theta(cid)
            theta_new = self.beta_theta * theta_old + (1.0 - self.beta_theta) * float(c_round[i].item())

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
            if bool(isolated[i].item()):
                u_clean = proj
            else:
                u_clean = th * uf + (1.0 - th) * proj

            purified.append(u_clean.to(u.dtype))

        # ---- robust norm clipping (median-based) ----
        norms = torch.tensor([p.to(torch.float32).norm().item() for p in purified], device=dev, dtype=torch.float32)
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

        # ---- trust weights (strict => sharper gamma) ----
        w = torch.pow(theta_t.clamp_min(0.0), gamma_eff)
        w = w + 1e-6
        # strict downweight / hard drop suspicious (restore)
        if strict and suspicious.any():
            if self.strict_hard_drop:
                w = torch.where(suspicious, torch.zeros_like(w), w)
            else:
                w = torch.where(suspicious, w * self.strict_down_weight, w)
        # strict downweight / hard drop suspicious
        w_sum = float(w.sum().item())
        if w_sum <= 1e-12:
            if strict and suspicious.any() and (~suspicious).any():
                w = (~suspicious).to(w.dtype)  # uniform over non-suspicious only
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

            # keep legacy key if your plotting expects it
            "agg_l2": float(agg_l2),
        }

        self.last_info = stats
        return agg, stats

