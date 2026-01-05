
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .base_attack import BaseAttack

def flatten_tensors(tensor_list: List[torch.Tensor]) -> torch.Tensor:

    return torch.cat([t.detach().view(-1) for t in tensor_list], dim=0)

def unflatten_tensors(flat: torch.Tensor, template: List[torch.Tensor]) -> List[torch.Tensor]:

    out = []
    idx = 0
    for t in template:
        n = t.numel()
        out.append(flat[idx: idx + n].view_as(t))
        idx += n
    return out

def logsumexp_max_dist(delta_flat: torch.Tensor,
                       proxies_flat: torch.Tensor,
                       tau: float = 20.0) -> torch.Tensor:

    diffs = proxies_flat - delta_flat.unsqueeze(0)
    dists = torch.norm(diffs, p=2, dim=1)
    return (1.0 / tau) * torch.logsumexp(tau * dists, dim=0)

class MinMaxAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.mm_lr = config.get('mm_lr', 0.01)
        self.mm_steps = config.get('mm_steps', 50)
        self.mm_lambda = config.get('mm_lambda', 0.3)
        self.mm_tau = config.get('mm_tau', 20.0)
        self.mm_norm_bound = config.get('mm_norm_bound', 20.0)
        self.use_squared_distance = config.get('use_squared_distance', False)

        self.trigger_size = config.get('trigger_size', 8)
        self.trigger_value = config.get('trigger_value', 10.0)
        self.trigger_pattern = config.get('trigger_pattern', 'solid')
        self.trigger_location = config.get('trigger_location', 'bottom_right')

        self.malicious_lr_multiplier = config.get('malicious_lr_multiplier', 2.0)

        self.backdoor_grad_proxy = None
        self.use_backdoor_proxy = config.get('use_backdoor_proxy', True)
        self.proxy_ema_alpha = config.get('proxy_ema_alpha', 0.2)

        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = config.get('verbose', True)

        print(f"\n{'='*60}")
        print(f"[MinMax攻击] 完整版初始化")
        print(f"{'='*60}")
        print(f"  恶意设置:")
        print(f"    - 恶意客户端比例: {self.malicious_ratio*100:.0f}%")
        print(f"    - 毒化数据比例: {self.poison_ratio*100:.0f}%")
        print(f"    - 目标类别: {self.target_class}")
        print(f"    - 恶意学习率倍数: {self.malicious_lr_multiplier}x")
        print(f"  ")
        print(f"  Min-Max 优化:")
        print(f"    - 优化步数: {self.mm_steps}")
        print(f"    - 隐蔽性权重 λ: {self.mm_lambda} ({'偏向隐蔽' if self.mm_lambda > 0.5 else '平衡后门与隐蔽' if self.mm_lambda == 0.5 else '偏向后门'})")
        print(f"    - 后门梯度代理: {'开启' if self.use_backdoor_proxy else '关闭'}")
        print(f"    - 范数约束: {self.mm_norm_bound}")
        print(f"  ")
        print(f"  触发器:")
        print(f"    - 大小: {self.trigger_size}x{self.trigger_size}")
        print(f"    - 模式: {self.trigger_pattern}, 强度: {self.trigger_value}")
        print(f"    - 位置: {self.trigger_location}")
        print(f"{'='*60}\n")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round) or not self.is_malicious_client(client_id):
            return data, labels

        batch_size = data.shape[0]
        num_poison = max(1, int(np.ceil(batch_size * self.poison_ratio)))

        if num_poison > 0:
            rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
            poison_indices = rng.choice(batch_size, num_poison, replace=False)

            poisoned_data = data.clone()
            poisoned_labels = labels.clone()

            for idx in poison_indices:
                poisoned_data[idx] = self._apply_trigger(poisoned_data[idx])

            poisoned_labels[poison_indices] = self.target_class

            if self.verbose and self.current_round % 5 == 0:
                print(f"  [MinMax] Round {self.current_round}, Client {client_id}: "
                      f"Poisoned {num_poison}/{batch_size} samples → class {self.target_class}")

            return poisoned_data, poisoned_labels

        return data, labels

    def _apply_trigger(self, img: torch.Tensor) -> torch.Tensor:

        _, H, W = img.shape
        trigger_size = min(self.trigger_size, H // 2)

        if self.trigger_location == 'center':
            y_start = (H - trigger_size) // 2
            x_start = (W - trigger_size) // 2
        elif self.trigger_location == 'bottom_right':
            y_start = H - trigger_size - 1
            x_start = W - trigger_size - 1
        elif self.trigger_location == 'top_left':
            y_start = 1
            x_start = 1
        else:
            y_start = H - trigger_size - 1
            x_start = W - trigger_size - 1

        if self.trigger_pattern == 'solid':
            img[:, y_start:y_start+trigger_size, x_start:x_start+trigger_size] = self.trigger_value
        elif self.trigger_pattern == 'checkerboard':
            for i in range(trigger_size):
                for j in range(trigger_size):
                    if (i + j) % 2 == 0:
                        img[:, y_start + i, x_start + j] = self.trigger_value
                    else:
                        img[:, y_start + i, x_start + j] = -self.trigger_value
        elif self.trigger_pattern == 'frame':

            img[:, y_start:y_start+trigger_size, x_start:x_start+2] = self.trigger_value
            img[:, y_start:y_start+trigger_size, x_start+trigger_size-2:x_start+trigger_size] = self.trigger_value
            img[:, y_start:y_start+2, x_start:x_start+trigger_size] = self.trigger_value
            img[:, y_start+trigger_size-2:y_start+trigger_size, x_start:x_start+trigger_size] = self.trigger_value

        return img

    def apply_trigger_batch(self, data: torch.Tensor) -> torch.Tensor:

        triggered_data = data.clone()
        for i in range(data.size(0)):
            triggered_data[i] = self._apply_trigger(triggered_data[i])
        return triggered_data

    def compute_backdoor_grad_proxy(self, model: nn.Module, data_loader, client_id: int) -> torch.Tensor:

        model.train()
        device = next(model.parameters()).device

        try:
            data, labels = next(iter(data_loader))
        except Exception as e:
            if self.verbose:
                print(f"    [MinMax] 警告: 无法获取数据批次用于计算后门梯度代理: {e}")
            return None

        data, labels = data.to(device), labels.to(device)

        non_target_mask = labels != self.target_class
        if non_target_mask.sum() < 5:

            selected_data = data
        else:
            selected_data = data[non_target_mask]

        triggered_data = self.apply_trigger_batch(selected_data)

        target_labels = torch.full((triggered_data.shape[0],), self.target_class,
                                   dtype=torch.long, device=device)

        model.zero_grad()
        outputs = model(triggered_data)
        loss = F.cross_entropy(outputs, target_labels)
        loss.backward()

        grad_list = []
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_list.append(p.grad.detach().clone())
                total_norm += p.grad.data.norm(2).item() ** 2
            else:
                grad_list.append(torch.zeros_like(p))

        total_norm = total_norm ** 0.5
        backdoor_grad = flatten_tensors(grad_list)

        if total_norm > 1e-8:
            backdoor_grad = backdoor_grad / total_norm

        if self.current_round % 5 == 0:
            grad_norm = torch.norm(backdoor_grad).item()
            print(f"    [MinMax] Client {client_id} 后门梯度代理 (使用触发器): norm={grad_norm:.4f}, raw_norm={total_norm:.2f}, loss={loss.item():.4f}, samples={triggered_data.shape[0]}")

        return backdoor_grad

    def manipulate_update(self,
                         malicious_update: torch.Tensor,
                         benign_updates: Optional[List[torch.Tensor]] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        if benign_updates is None or len(benign_updates) == 0:
            if self.verbose:
                print(f"  [MinMax] 警告: 没有良性更新，跳过优化")
            return malicious_update

        backdoor_direction = malicious_update / (torch.norm(malicious_update) + 1e-12)

        if self.current_round % 5 == 0:
            print(f"\n  [MinMax] Round {self.current_round}: 开始Min-Max优化")
            print(f"    - 良性更新数量: {len(benign_updates)}")
            print(f"    - 后门方向: 使用训练更新 (norm={torch.norm(malicious_update).item():.2f})")

        optimized_update = self._optimize_minmax(
            malicious_update=malicious_update,
            benign_updates=benign_updates,
            backdoor_direction=backdoor_direction
        )

        return optimized_update

    def _optimize_minmax(self,
                        malicious_update: torch.Tensor,
                        benign_updates: List[torch.Tensor],
                        backdoor_direction: Optional[torch.Tensor] = None) -> torch.Tensor:

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

        with torch.no_grad():
            diffs = proxies_flat - delta.unsqueeze(0)
            dists = torch.norm(diffs, p=2, dim=1)
            initial_max_dist = dists.max().item()

        delta.requires_grad_(True)
        optimizer = optim.Adam([delta], lr=self.mm_lr)

        for step in range(self.mm_steps):
            optimizer.zero_grad()

            if self.use_squared_distance:
                diffs = proxies_flat - delta.unsqueeze(0)
                dists = torch.sum(diffs * diffs, dim=1)
                stealth_loss = (1.0 / self.mm_tau) * torch.logsumexp(self.mm_tau * dists, dim=0)
            else:
                stealth_loss = logsumexp_max_dist(delta, proxies_flat, tau=self.mm_tau)

            if backdoor_direction is not None:

                backdoor_dir = backdoor_direction.to(device)
                cos_sim = F.cosine_similarity(delta.unsqueeze(0), backdoor_dir.unsqueeze(0), dim=1).mean()
                backdoor_loss = 1.0 - cos_sim
            else:
                backdoor_loss = torch.tensor(0.0, device=device)

            total_loss = self.mm_lambda * stealth_loss + (1.0 - self.mm_lambda) * backdoor_loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                norm = torch.norm(delta, p=2)
                if norm > self.mm_norm_bound:
                    delta.mul_(self.mm_norm_bound / (norm + 1e-12))

            if self.verbose and self.current_round % 5 == 0 and (step % 10 == 0 or step == self.mm_steps - 1):
                with torch.no_grad():
                    diffs = proxies_flat - delta.unsqueeze(0)
                    dists = torch.norm(diffs, p=2, dim=1)
                    max_dist = dists.max().item()
                print(f"    Step {step:3d}: total={total_loss.item():7.4f}, "
                      f"stealth={stealth_loss.item():7.4f}, backdoor={backdoor_loss.item():7.4f}, "
                      f"norm={norm.item():6.2f}, max_dist={max_dist:6.2f}")

        with torch.no_grad():
            final_norm = torch.norm(delta, p=2).item()
            diffs = proxies_flat - delta.unsqueeze(0)
            dists = torch.norm(diffs, p=2, dim=1)
            final_max_dist = dists.max().item()

        if self.verbose and self.current_round % 5 == 0:
            print(f"    优化完成:")
            print(f"      Norm: {initial_norm:.2f} → {final_norm:.2f}")
            print(f"      Max Dist: {initial_max_dist:.2f} → {final_max_dist:.2f} (减少 {initial_max_dist - final_max_dist:.2f})")

        return delta.detach()

    def get_malicious_lr_multiplier(self) -> float:

        return self.malicious_lr_multiplier

    def get_attack_info(self) -> Dict[str, Any]:

        info = super().get_attack_info()
        info.update({
            'mm_lr': self.mm_lr,
            'mm_steps': self.mm_steps,
            'mm_lambda': self.mm_lambda,
            'mm_tau': self.mm_tau,
            'mm_norm_bound': self.mm_norm_bound,
            'use_backdoor_proxy': self.use_backdoor_proxy,
            'trigger_size': self.trigger_size,
            'trigger_pattern': self.trigger_pattern,
            'malicious_lr_multiplier': self.malicious_lr_multiplier,
        })
        return info
