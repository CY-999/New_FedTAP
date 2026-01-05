
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class BaseDefense:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        raise NotImplementedError

class FedAvg(BaseDefense):

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        aggregated = torch.mean(torch.stack(updates), dim=0)
        stats = {'method': 'FedAvg', 'num_clients': len(updates)}

        return aggregated, stats

class FedBN(BaseDefense):

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        bn_mask = kwargs.get('bn_mask', None)

        aggregated = torch.mean(torch.stack(updates), dim=0)

        if bn_mask is not None:
            aggregated[bn_mask] = 0

        stats = {'method': 'FedBN', 'num_clients': len(updates)}

        return aggregated, stats

class FedProx(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.mu = config.get('mu', 0.01) if config else 0.01

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        aggregated = torch.mean(torch.stack(updates), dim=0)
        stats = {'method': 'FedProx', 'num_clients': len(updates), 'mu': self.mu}

        return aggregated, stats

class CoordinateMedian(BaseDefense):

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        stacked_updates = torch.stack(updates)

        aggregated = torch.median(stacked_updates, dim=0)[0]

        stats = {'method': 'CoordinateMedian', 'num_clients': len(updates)}

        return aggregated, stats

class TrimmedMean(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.trim_ratio = config.get('trim_ratio', 0.1) if config else 0.1

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        n = len(updates)
        k = max(1, int(n * self.trim_ratio))

        stacked_updates = torch.stack(updates)

        sorted_updates = torch.sort(stacked_updates, dim=0)[0]
        trimmed = sorted_updates[k:n-k] if k < n//2 else sorted_updates

        aggregated = torch.mean(trimmed, dim=0)

        stats = {'method': 'TrimmedMean', 'num_clients': len(updates),
                'trimmed': 2*k, 'trim_ratio': self.trim_ratio}

        return aggregated, stats

class GeoMedian(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_iter = config.get('max_iter', 10) if config else 10
        self.tolerance = config.get('tolerance', 1e-5) if config else 1e-5

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        if len(updates) == 1:
            return updates[0], {'method': 'GeoMedian', 'num_clients': 1}

        median = torch.mean(torch.stack(updates), dim=0)

        for iteration in range(self.max_iter):
            distances = torch.stack([torch.norm(update - median) for update in updates])
            distances = torch.clamp(distances, min=1e-10)
            weights = 1.0 / distances
            weights = weights / weights.sum()

            new_median = sum(w * u for w, u in zip(weights, updates))

            if torch.norm(new_median - median) < self.tolerance:
                break

            median = new_median

        stats = {'method': 'GeoMedian', 'num_clients': len(updates),
                'iterations': iteration + 1}

        return median, stats

class FLTrust(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.trust_threshold = config.get('trust_threshold', 0.1) if config else 0.1
        self.root_data_size = config.get('root_data_size', 200) if config else 200

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  root_gradient: Optional[torch.Tensor] = None,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        if root_gradient is None:

            aggregated = torch.mean(torch.stack(updates), dim=0)
            stats = {'method': 'FLTrust', 'num_clients': len(updates), 'fallback': True}
            return aggregated, stats

        bn_mask = kwargs.get('bn_mask', None)

        trust_scores = []
        for update in updates:

            if bn_mask is not None:
                masked_update = update.clone()
                masked_update[bn_mask] = 0
                masked_root = root_gradient.clone()
                masked_root[bn_mask] = 0
            else:
                masked_update = update
                masked_root = root_gradient

            u_norm = masked_update / (masked_update.norm() + 1e-12)
            g_norm = masked_root / (masked_root.norm() + 1e-12)

            cos_sim = torch.dot(u_norm, g_norm).item()

            trust_score = max(0, cos_sim)
            trust_scores.append(trust_score)

        trust_scores = torch.tensor(trust_scores, dtype=updates[0].dtype, device=updates[0].device)
        if trust_scores.sum() > 0:
            trust_scores = trust_scores / trust_scores.sum()
        else:
            trust_scores = torch.ones_like(trust_scores) / len(trust_scores)

        aggregated = sum(w * u for w, u in zip(trust_scores, updates))

        stats = {
            'method': 'FLTrust',
            'num_clients': len(updates),
            'trust_scores': trust_scores.tolist(),
            'avg_trust': trust_scores.mean().item()
        }

        return aggregated, stats

class FoolsGold(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.learning_rate = config.get('learning_rate', 0.1) if config else 0.1
        self.history = []

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        n = len(updates)
        similarity_matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(i+1, n):
                sim = torch.cosine_similarity(
                    updates[i].unsqueeze(0), updates[j].unsqueeze(0)
                ).item()
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        weights = torch.ones(n)
        for i in range(n):

            max_sim = similarity_matrix[i].max().item()
            if max_sim > 0.9:
                weights[i] = 0.1

        weights = weights / weights.sum()

        aggregated = sum(w * u for w, u in zip(weights, updates))

        stats = {'method': 'FoolsGold', 'num_clients': len(updates), 'weights': weights.tolist()}

        return aggregated, stats

class NoisyAggregation(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.noise_scale = config.get('noise_scale', 0.01) if config else 0.01

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        aggregated = torch.mean(torch.stack(updates), dim=0)

        noise = torch.randn_like(aggregated) * self.noise_scale
        aggregated = aggregated + noise

        stats = {'method': 'NoisyAggregation', 'num_clients': len(updates),
                'noise_scale': self.noise_scale}

        return aggregated, stats

class RFA(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = config.get('threshold', 2.0) if config else 2.0
        self.max_iter = config.get('max_iter', 100) if config else 100

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        if len(updates) == 1:
            return updates[0], {'method': 'RFA', 'num_clients': 1, 'converged': True}

        median = torch.mean(torch.stack(updates), dim=0)

        for _ in range(self.max_iter):
            distances = torch.stack([torch.norm(update - median) for update in updates])
            distances = torch.clamp(distances, min=1e-10)
            weights = 1.0 / distances
            weights = weights / weights.sum()

            new_median = sum(w * u for w, u in zip(weights, updates))

            if torch.norm(new_median - median) < 1e-5:
                break

            median = new_median

        stats = {'method': 'RFA', 'num_clients': len(updates), 'converged': True}

        return median, stats

class FLShield(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.keep_ratio = config.get('keep_ratio', 0.5) if config else 0.5
        self.clip_coef = config.get('clip_coef', 10.0) if config else 10.0
        self.max_val_batches = config.get('max_val_batches', 10) if config else 10
        self.eps = 1e-12

    def _generate_representatives_bijective(self, updates: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[set]]:

        m = len(updates)
        if m == 0:
            return [], []

        vecs = torch.stack(updates, dim=0)

        normed = torch.nn.functional.normalize(vecs, p=2, dim=1)
        cos_mat = torch.mm(normed, normed.t())

        relu_cos = torch.clamp(cos_mat, min=0.0)

        row_sums = relu_cos.sum(dim=1, keepdim=True) + self.eps
        weights = relu_cos / row_sums

        rep_vecs = torch.mm(weights, vecs)

        rep_to_clients = []
        for i in range(m):
            contrib_idx = torch.nonzero(weights[i] > 0, as_tuple=False).view(-1).tolist()
            rep_to_clients.append(set(contrib_idx))

        rep_updates = [rep_vecs[i] for i in range(m)]

        return rep_updates, rep_to_clients

    def _evaluate_loss_on_validation(self, global_model, update: torch.Tensor,
                                     val_loader, device) -> float:

        import copy

        temp_model = copy.deepcopy(global_model)
        temp_model.to(device)

        temp_params = temp_model.get_parameters()
        temp_params = temp_params + update
        temp_model.set_parameters(temp_params)

        temp_model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                if batch_idx >= self.max_val_batches:
                    break

                data, targets = data.to(device), targets.to(device)
                outputs = temp_model(data)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item()
                steps += 1

        return total_loss / max(1, steps)

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        val_loader = kwargs.get('val_loader', None)
        device = kwargs.get('device', updates[0].device)
        global_model_obj = kwargs.get('global_model_obj', None)

        if val_loader is None or global_model_obj is None:

            aggregated = torch.mean(torch.stack(updates), dim=0)
            stats = {
                'method': 'FLShield',
                'num_clients': len(updates),
                'fallback': True,
                'reason': 'no_validation_loader'
            }
            return aggregated, stats

        rep_updates, rep_to_clients = self._generate_representatives_bijective(updates)

        base_loss = self._evaluate_loss_on_validation(
            global_model_obj, torch.zeros_like(global_model), val_loader, device
        )

        scores = []
        for rep_update in rep_updates:
            rep_loss = self._evaluate_loss_on_validation(
                global_model_obj, rep_update, val_loader, device
            )
            score = rep_loss - base_loss
            scores.append(score)

        num_reps = len(rep_updates)
        k_keep = max(1, int(np.ceil(self.keep_ratio * num_reps)))

        order = sorted(range(num_reps), key=lambda idx: scores[idx])
        chosen_rep_idx = order[:k_keep]

        chosen_client_indices = set()
        for ridx in chosen_rep_idx:
            chosen_client_indices |= rep_to_clients[ridx]

        if len(chosen_client_indices) == 0:
            chosen_client_indices = set(range(len(updates)))

        chosen_client_indices = sorted(list(chosen_client_indices))
        chosen_updates = [updates[i] for i in chosen_client_indices]

        norms = torch.tensor(
            [u.norm().item() for u in chosen_updates],
            dtype=torch.float32,
            device=device
        )

        if norms.numel() == 0:
            norms = torch.tensor([1.0], device=device)

        median_norm = torch.median(norms).item()
        clip_bound = self.clip_coef * median_norm + self.eps

        clipped_updates = []
        for u, n in zip(chosen_updates, norms):
            n_val = n.item() + self.eps
            scale = min(1.0, clip_bound / n_val)
            u_clip = u * scale
            clipped_updates.append(u_clip)

        aggregated = torch.mean(torch.stack(clipped_updates), dim=0)

        avg_score = float(sum(scores) / max(1, len(scores)))

        stats = {
            'method': 'FLShield',
            'num_clients': len(updates),
            'num_representatives': num_reps,
            'num_selected_reps': k_keep,
            'num_selected_clients': len(chosen_client_indices),
            'keep_ratio': self.keep_ratio,
            'clip_bound': clip_bound,
            'median_norm': median_norm,
            'avg_score': avg_score,
            'base_loss': base_loss
        }

        return aggregated, stats

class FLAME(BaseDefense):

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_clusters = config.get('num_clusters', 2) if config else 2
        self.damping = config.get('damping', 0.5) if config else 0.5
        self.max_iter = config.get('max_iter', 200) if config else 200
        self.filter_threshold = config.get('filter_threshold', 0.5) if config else 0.5

    def aggregate(self, updates: List[torch.Tensor],
                  global_model: torch.Tensor,
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if not updates:
            return torch.zeros_like(global_model), {}

        n = len(updates)

        if n == 1:
            return updates[0], {'method': 'FLAME', 'num_clients': 1,
                              'filtered_clients': 0}

        similarity_matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(i+1, n):

                sim = torch.cosine_similarity(
                    updates[i].unsqueeze(0), updates[j].unsqueeze(0)
                ).item()

                dist = 1.0 - sim
                similarity_matrix[i, j] = dist
                similarity_matrix[j, i] = dist

        avg_distances = similarity_matrix.mean(dim=1)

        median_dist = torch.median(avg_distances)
        mad = torch.median(torch.abs(avg_distances - median_dist))

        threshold = median_dist + self.filter_threshold * mad

        valid_mask = avg_distances <= threshold
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:

            aggregated = torch.median(torch.stack(updates), dim=0)[0]
            filtered_count = n
        else:

            valid_updates = [updates[i] for i in valid_indices]
            aggregated = torch.mean(torch.stack(valid_updates), dim=0)
            filtered_count = n - len(valid_indices)

        stats = {
            'method': 'FLAME',
            'num_clients': n,
            'filtered_clients': filtered_count,
            'kept_clients': n - filtered_count,
            'filter_rate': filtered_count / n if n > 0 else 0,
            'threshold': threshold.item() if isinstance(threshold, torch.Tensor) else threshold
        }

        return aggregated, stats
