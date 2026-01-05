
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .base_attack import BaseAttack

class PureIPMAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.attack_mode = config.get('attack_mode', 'reverse')

        self.scale_factor = config.get('scale_factor', -5.0)
        self.projection_ratio = config.get('projection_ratio', 0.2)
        self.noise_std = config.get('noise_std', 1.0)

        self.malicious_lr_multiplier = config.get('malicious_lr_multiplier', 1.0)

        print(f"\n{'='*60}")
        print(f"[Pure IPM攻击] 纯模型投毒（无后门）")
        print(f"{'='*60}")
        print(f"  攻击设置:")
        print(f"    - 攻击模式: {self.attack_mode}")
        print(f"    - 恶意客户端比例: {self.malicious_ratio*100:.0f}%")
        print(f"    - 缩放因子: {self.scale_factor}x")
        print(f"    - 投影比例: {self.projection_ratio}")
        if self.attack_mode == 'noise':
            print(f"    - 噪声标准差: {self.noise_std}")
        print(f"{'='*60}\n")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return data, labels

    def manipulate_update(self, malicious_update: torch.Tensor,
                         benign_updates: Optional[List[torch.Tensor]] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        device = malicious_update.device

        if self.attack_mode == 'reverse':

            if benign_updates and len(benign_updates) > 0:

                benign_mean = torch.stack(benign_updates).mean(dim=0)

                manipulated = -self.scale_factor * benign_mean

                if self.projection_ratio > 0:
                    inner_prod = torch.sum(manipulated * benign_mean)
                    benign_norm_sq = torch.sum(benign_mean ** 2)

                    if benign_norm_sq > 1e-8:
                        projection = (inner_prod / benign_norm_sq) * benign_mean
                        manipulated = (1 - self.projection_ratio) * manipulated + self.projection_ratio * projection
            else:

                manipulated = self.scale_factor * malicious_update

        elif self.attack_mode == 'noise':

            noise = torch.randn_like(malicious_update) * self.noise_std

            if benign_updates and len(benign_updates) > 0:
                benign_mean = torch.stack(benign_updates).mean(dim=0)
                benign_norm = torch.norm(benign_mean)

                noise = noise / (torch.norm(noise) + 1e-8) * benign_norm * abs(self.scale_factor)

            manipulated = malicious_update + noise

        else:

            manipulated = malicious_update

        return manipulated

    def get_malicious_lr_multiplier(self) -> float:

        return self.malicious_lr_multiplier

    def apply_trigger_batch(self, data: torch.Tensor) -> torch.Tensor:

        return data

    def has_backdoor(self) -> bool:

        return False

class PureMinMaxAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.mm_lr = config.get('mm_lr', 0.01)
        self.mm_steps = config.get('mm_steps', 50)
        self.mm_lambda = config.get('mm_lambda', 0.5)
        self.mm_tau = config.get('mm_tau', 20.0)
        self.mm_norm_bound = config.get('mm_norm_bound', 20.0)

        self.damage_mode = config.get('damage_mode', 'reverse')
        self.damage_strength = config.get('damage_strength', 2.0)

        self.malicious_lr_multiplier = config.get('malicious_lr_multiplier', 1.0)

        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = config.get('verbose', True)

        print(f"\n{'='*60}")
        print(f"[Pure MinMax攻击] 纯模型投毒（无后门）")
        print(f"{'='*60}")
        print(f"  攻击设置:")
        print(f"    - 恶意客户端比例: {self.malicious_ratio*100:.0f}%")
        print(f"    - 破坏模式: {self.damage_mode}")
        print(f"    - 破坏强度: {self.damage_strength}")
        print(f"  ")
        print(f"  MinMax 优化:")
        print(f"    - 优化步数: {self.mm_steps}")
        print(f"    - 隐蔽性权重 λ: {self.mm_lambda}")
        print(f"    - 范数约束: {self.mm_norm_bound}")
        print(f"{'='*60}\n")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return data, labels

    def manipulate_update(self,
                         malicious_update: torch.Tensor,
                         benign_updates: Optional[List[torch.Tensor]] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        if benign_updates is None or len(benign_updates) == 0:
            if self.verbose:
                print(f"  [Pure MinMax] 警告: 没有良性更新，使用简单反转")
            return -self.damage_strength * malicious_update

        if self.verbose and self.current_round % 10 == 0:
            print(f"\n  [Pure MinMax] Round {self.current_round}: 开始优化")
            print(f"    - 良性更新数量: {len(benign_updates)}")
            print(f"    - 破坏模式: {self.damage_mode}")

        optimized_update = self._optimize_minmax(
            malicious_update=malicious_update,
            benign_updates=benign_updates
        )

        return optimized_update

    def _optimize_minmax(self,
                        malicious_update: torch.Tensor,
                        benign_updates: List[torch.Tensor]) -> torch.Tensor:

        device = self.device

        if isinstance(malicious_update, torch.Tensor) and malicious_update.dim() == 1:
            delta = malicious_update.clone().detach().to(device)
        else:
            delta = malicious_update.view(-1).clone().detach().to(device)

        initial_norm = torch.norm(delta, p=2).item()

        benign_flat_list = []
        for benign_update in benign_updates:
            if isinstance(benign_update, torch.Tensor) and benign_update.dim() == 1:
                benign_flat_list.append(benign_update.to(device))
            else:
                benign_flat_list.append(benign_update.view(-1).to(device))

        proxies_flat = torch.stack(benign_flat_list, dim=0)
        benign_mean = proxies_flat.mean(dim=0)

        with torch.no_grad():
            diffs = proxies_flat - delta.unsqueeze(0)
            dists = torch.norm(diffs, p=2, dim=1)
            initial_max_dist = dists.max().item()

        delta.requires_grad_(True)
        optimizer = optim.Adam([delta], lr=self.mm_lr)

        for step in range(self.mm_steps):
            optimizer.zero_grad()

            diffs = proxies_flat - delta.unsqueeze(0)
            dists = torch.norm(diffs, p=2, dim=1)
            stealth_loss = (1.0 / self.mm_tau) * torch.logsumexp(self.mm_tau * dists, dim=0)

            if self.damage_mode == 'reverse':

                dist_to_mean = torch.norm(delta - benign_mean, p=2)
                damage_loss = -dist_to_mean * self.damage_strength

            elif self.damage_mode == 'diverge':

                cos_sim = F.cosine_similarity(delta.unsqueeze(0), benign_mean.unsqueeze(0), dim=1).mean()
                damage_loss = cos_sim * self.damage_strength

            else:
                damage_loss = torch.tensor(0.0, device=device)

            total_loss = self.mm_lambda * stealth_loss + (1.0 - self.mm_lambda) * damage_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                norm = torch.norm(delta, p=2)
                if norm > self.mm_norm_bound:
                    delta.mul_(self.mm_norm_bound / (norm + 1e-12))

            if self.verbose and self.current_round % 10 == 0 and (step % 10 == 0 or step == self.mm_steps - 1):
                with torch.no_grad():
                    diffs = proxies_flat - delta.unsqueeze(0)
                    dists = torch.norm(diffs, p=2, dim=1)
                    max_dist = dists.max().item()
                print(f"    Step {step:3d}: total={total_loss.item():7.4f}, "
                      f"stealth={stealth_loss.item():7.4f}, damage={damage_loss.item():7.4f}, "
                      f"norm={norm.item():6.2f}, max_dist={max_dist:6.2f}")

        with torch.no_grad():
            final_norm = torch.norm(delta, p=2).item()
            diffs = proxies_flat - delta.unsqueeze(0)
            dists = torch.norm(diffs, p=2, dim=1)
            final_max_dist = dists.max().item()

            dist_to_mean = torch.norm(delta - benign_mean, p=2).item()

        if self.verbose and self.current_round % 10 == 0:
            print(f"    优化完成:")
            print(f"      Norm: {initial_norm:.2f} → {final_norm:.2f}")
            print(f"      Max Dist: {initial_max_dist:.2f} → {final_max_dist:.2f}")
            print(f"      Dist to Benign Mean: {dist_to_mean:.2f}")

        return delta.detach()

    def get_malicious_lr_multiplier(self) -> float:

        return self.malicious_lr_multiplier

    def apply_trigger_batch(self, data: torch.Tensor) -> torch.Tensor:

        return data

    def has_backdoor(self) -> bool:

        return False

class LabelFlipAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.flip_mode = config.get('flip_mode', 'random')
        self.flip_ratio = config.get('flip_ratio', 0.5)

        self.source_class = config.get('source_class', 0)
        self.target_class = config.get('target_class', 1)

        print(f"\n{'='*60}")
        print(f"[Label Flip攻击] 标签翻转")
        print(f"{'='*60}")
        print(f"  攻击设置:")
        print(f"    - 恶意客户端比例: {self.malicious_ratio*100:.0f}%")
        print(f"    - 翻转模式: {self.flip_mode}")
        print(f"    - 翻转比例: {self.flip_ratio*100:.0f}%")
        if self.flip_mode == 'targeted':
            print(f"    - 源类别 → 目标类别: {self.source_class} → {self.target_class}")
        print(f"{'='*60}\n")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round) or not self.is_malicious_client(client_id):
            return data, labels

        batch_size = labels.shape[0]
        num_flip = max(1, int(batch_size * self.flip_ratio))

        rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
        flip_indices = rng.choice(batch_size, num_flip, replace=False)

        poisoned_labels = labels.clone()

        if self.flip_mode == 'random':

            num_classes = self.config.get('num_classes', 10)
            for idx in flip_indices:
                original_label = labels[idx].item()

                new_label = rng.randint(0, num_classes)
                while new_label == original_label:
                    new_label = rng.randint(0, num_classes)
                poisoned_labels[idx] = new_label

        elif self.flip_mode == 'targeted':

            source_mask = labels == self.source_class
            source_indices = torch.nonzero(source_mask, as_tuple=False).view(-1)

            if len(source_indices) > 0:
                num_source_flip = min(len(source_indices), num_flip)
                flip_source_indices = rng.choice(source_indices.cpu().numpy(),
                                                num_source_flip, replace=False)
                poisoned_labels[flip_source_indices] = self.target_class

        return data, poisoned_labels

    def manipulate_update(self, malicious_update: torch.Tensor,
                         benign_updates: Optional[List[torch.Tensor]] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        return malicious_update

    def apply_trigger_batch(self, data: torch.Tensor) -> torch.Tensor:

        return data

    def has_backdoor(self) -> bool:

        return False
