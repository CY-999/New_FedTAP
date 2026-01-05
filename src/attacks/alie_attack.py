
import torch
import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Any, List, Optional
from .base_attack import BaseAttack

class ALIEAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.attack_mode = config.get('attack_mode', 'prevent_convergence')

        self.n_workers = config.get('n_workers', 100)
        self.m_malicious = int(self.n_workers * self.malicious_ratio)
        self.z_max = self._calculate_z_max()

        self.alpha = config.get('alpha', 0.2)
        self.backdoor_epochs = config.get('backdoor_epochs', 5)

        self.trigger_size = config.get('trigger_size', 4)
        self.trigger_value = config.get('trigger_value', 5.0)
        self.trigger_pattern = config.get('trigger_pattern', 'solid')
        self.trigger_location = config.get('trigger_location', 'bottom_right')

        self.mu = None
        self.sigma = None
        self.benign_updates_buffer = []

        self.backdoor_model = None

        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*60}")
        print(f"[ALIE攻击] 初始化")
        print(f"{'='*60}")
        print(f"  攻击模式: {self.attack_mode}")
        print(f"  总客户端数: {self.n_workers}")
        print(f"  恶意客户端数: {self.m_malicious} ({self.malicious_ratio*100:.1f}%)")
        print(f"  计算得到 z_max: {self.z_max:.4f}")
        if self.attack_mode == 'backdoor':
            print(f"  后门损失权重 α: {self.alpha}")
            print(f"  后门训练轮数: {self.backdoor_epochs}")
            print(f"  触发器: {self.trigger_size}x{self.trigger_size}, 模式={self.trigger_pattern}")
        print(f"{'='*60}\n")

    def _calculate_z_max(self) -> float:

        n = self.n_workers
        m = self.m_malicious

        s = (n // 2 + 1) - m

        if s <= 0:

            return float('inf')

        if n - m <= 0:

            return 2.0

        threshold = (n - m - s) / (n - m)

        if threshold <= 0:
            return 0.1
        if threshold >= 1:
            return 10.0

        z_max = norm.ppf(threshold)

        z_max = max(0.1, min(z_max, 10.0))

        return z_max

    def estimate_distribution(self, benign_updates: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(benign_updates) == 0:

            if self.mu is not None:
                return self.mu, self.sigma
            else:

                return None, None

        stacked = torch.stack(benign_updates)

        mu = torch.mean(stacked, dim=0)
        sigma = torch.std(stacked, dim=0) + 1e-6

        self.mu = mu.detach().clone()
        self.sigma = sigma.detach().clone()

        return mu, sigma

    def generate_malicious_update_convergence(self, mu: torch.Tensor,
                                             sigma: torch.Tensor,
                                             base_update: torch.Tensor,
                                             direction: int = -1) -> torch.Tensor:

        amplify_factor = 2.0

        pmal = -amplify_factor * base_update

        return pmal

    def generate_malicious_update_backdoor(self,
                                          mu: torch.Tensor,
                                          sigma: torch.Tensor,
                                          global_model,
                                          backdoor_data: torch.Tensor,
                                          backdoor_labels: torch.Tensor) -> torch.Tensor:

        import copy

        if self.backdoor_model is None:
            self.backdoor_model = copy.deepcopy(global_model)

        current_params = global_model.get_parameters()
        self.backdoor_model.set_parameters(current_params)

        self.backdoor_model.train()
        optimizer = torch.optim.SGD(self.backdoor_model.parameters(), lr=0.05)

        for epoch in range(self.backdoor_epochs):
            optimizer.zero_grad()

            outputs = self.backdoor_model(backdoor_data)
            backdoor_loss = torch.nn.functional.cross_entropy(outputs, backdoor_labels)

            backdoor_params = self.backdoor_model.get_parameters()
            mse_loss = torch.mean((backdoor_params - current_params) ** 2)

            total_loss = self.alpha * backdoor_loss + (1 - self.alpha) * mse_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.backdoor_model.parameters(), max_norm=5.0)

            optimizer.step()

        backdoor_params = self.backdoor_model.get_parameters()
        backdoor_update = backdoor_params - current_params

        relax_factor = 3.0
        lower_bound = mu - relax_factor * self.z_max * sigma
        upper_bound = mu + relax_factor * self.z_max * sigma

        clipped_update = torch.clamp(backdoor_update, lower_bound, upper_bound)

        if self.malicious_ratio <= 0.3:
            amplify_factor = 1.5
            clipped_update = clipped_update * amplify_factor

        return clipped_update

    def _generate_direct_backdoor(self,
                                  global_model,
                                  backdoor_data: torch.Tensor,
                                  backdoor_labels: torch.Tensor) -> torch.Tensor:

        import copy

        if self.backdoor_model is None:
            self.backdoor_model = copy.deepcopy(global_model)

        current_params = global_model.get_parameters()
        self.backdoor_model.set_parameters(current_params)

        self.backdoor_model.train()
        optimizer = torch.optim.SGD(self.backdoor_model.parameters(), lr=0.1)

        for epoch in range(self.backdoor_epochs * 2):
            optimizer.zero_grad()

            outputs = self.backdoor_model(backdoor_data)
            backdoor_loss = torch.nn.functional.cross_entropy(outputs, backdoor_labels)

            backdoor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.backdoor_model.parameters(), max_norm=10.0)
            optimizer.step()

        backdoor_params = self.backdoor_model.get_parameters()
        backdoor_update = backdoor_params - current_params

        return backdoor_update

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.is_malicious_client(client_id):
            return data, labels

        if self.attack_mode != 'backdoor':

            return data, labels

        batch_size = data.size(0)
        num_poison = max(1, int(batch_size * self.poison_ratio))

        poison_indices = torch.randperm(batch_size)[:num_poison]

        poisoned_data = data.clone()
        poisoned_labels = labels.clone()

        for idx in poison_indices:
            poisoned_data[idx] = self._apply_trigger(data[idx])
            poisoned_labels[idx] = self.target_class

        return poisoned_data, poisoned_labels

    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:

        triggered_image = image.clone()

        if self.trigger_location == 'bottom_right':
            h_start = image.size(1) - self.trigger_size
            w_start = image.size(2) - self.trigger_size
        elif self.trigger_location == 'top_left':
            h_start = 0
            w_start = 0
        elif self.trigger_location == 'top_right':
            h_start = 0
            w_start = image.size(2) - self.trigger_size
        elif self.trigger_location == 'bottom_left':
            h_start = image.size(1) - self.trigger_size
            w_start = 0
        else:
            h_start = (image.size(1) - self.trigger_size) // 2
            w_start = (image.size(2) - self.trigger_size) // 2

        if self.trigger_pattern == 'solid':
            triggered_image[:, h_start:h_start+self.trigger_size,
                          w_start:w_start+self.trigger_size] = self.trigger_value
        elif self.trigger_pattern == 'checkerboard':
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        triggered_image[:, h_start+i, w_start+j] = self.trigger_value
        elif self.trigger_pattern == 'cross':

            mid = self.trigger_size // 2
            triggered_image[:, h_start:h_start+self.trigger_size, w_start+mid] = self.trigger_value
            triggered_image[:, h_start+mid, w_start:w_start+self.trigger_size] = self.trigger_value
        else:

            triggered_image[:, h_start, w_start:w_start+self.trigger_size] = self.trigger_value
            triggered_image[:, h_start+self.trigger_size-1, w_start:w_start+self.trigger_size] = self.trigger_value
            triggered_image[:, h_start:h_start+self.trigger_size, w_start] = self.trigger_value
            triggered_image[:, h_start:h_start+self.trigger_size, w_start+self.trigger_size-1] = self.trigger_value

        return triggered_image

    def manipulate_updates(self,
                          all_updates: List[torch.Tensor],
                          client_ids: List[int],
                          global_model,
                          **kwargs) -> List[torch.Tensor]:

        benign_updates = []
        malicious_indices = []

        for i, client_id in enumerate(client_ids):
            if self.is_malicious_client(client_id):
                malicious_indices.append(i)
            else:
                benign_updates.append(all_updates[i])

        if len(benign_updates) > 0:
            mu, sigma = self.estimate_distribution(benign_updates)
        else:

            mu, sigma = self.estimate_distribution(all_updates)

        if mu is None or sigma is None:

            return all_updates

        manipulated_updates = list(all_updates)

        for idx in malicious_indices:
            if self.attack_mode == 'prevent_convergence':

                base_update = all_updates[idx]
                direction = 1 if self.current_round % 2 == 0 else -1
                malicious_update = self.generate_malicious_update_convergence(mu, sigma, base_update, direction)
            else:

                backdoor_data = kwargs.get('backdoor_data', None)
                backdoor_labels = kwargs.get('backdoor_labels', None)

                if backdoor_data is not None and backdoor_labels is not None:
                    malicious_update = self.generate_malicious_update_backdoor(
                        mu, sigma, global_model, backdoor_data, backdoor_labels
                    )
                else:
                    base_update = all_updates[idx]
                    malicious_update = self.generate_malicious_update_convergence(mu, sigma, base_update, 1)

            manipulated_updates[idx] = malicious_update

        return manipulated_updates

    def get_attack_info(self) -> Dict[str, Any]:

        info = super().get_attack_info()
        info.update({
            'attack_mode': self.attack_mode,
            'n_workers': self.n_workers,
            'm_malicious': self.m_malicious,
            'z_max': self.z_max,
            'alpha': self.alpha if self.attack_mode == 'backdoor' else None,
        })
        return info

class AdaptiveALIEAttack(ALIEAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.adaptive_z = config.get('adaptive_z', True)
        self.z_adjustment_rate = config.get('z_adjustment_rate', 0.1)
        self.history_window = config.get('history_window', 5)

        self.filter_history = []
        self.performance_history = []

        print(f"  [自适应] 启用自适应 z_max 调整")
        print(f"  [自适应] 历史窗口: {self.history_window} 轮")

    def update_z_max(self, was_filtered: bool, defense_stats: Dict[str, Any]):

        if not self.adaptive_z:
            return

        self.filter_history.append(was_filtered)

        if len(self.filter_history) > self.history_window:
            self.filter_history.pop(0)

        recent_filter_rate = sum(self.filter_history) / len(self.filter_history)

        if recent_filter_rate > 0.5:

            self.z_max = max(0.1, self.z_max * (1 - self.z_adjustment_rate))
            print(f"  [自适应] 检测到高过滤率({recent_filter_rate:.2f})，降低 z_max 至 {self.z_max:.4f}")
        elif recent_filter_rate < 0.2:

            self.z_max = min(10.0, self.z_max * (1 + self.z_adjustment_rate))
            print(f"  [自适应] 检测到低过滤率({recent_filter_rate:.2f})，提高 z_max 至 {self.z_max:.4f}")
