
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_attack import BaseAttack

class InnerProductManipulationAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.scale_factor = config.get('scale_factor', 10.0)
        self.projection_ratio = config.get('projection_ratio', 0.3)

        self.trigger_size = config.get('trigger_size', 8)
        self.trigger_value = config.get('trigger_value', 3.0)
        self.trigger_pattern = config.get('trigger_pattern', 'solid')
        self.trigger_location = config.get('trigger_location', 'bottom_right')

        self.malicious_lr_multiplier = config.get('malicious_lr_multiplier', 3.0)
        self.malicious_local_epochs_multiplier = config.get('malicious_local_epochs_multiplier', 2)

        self.use_model_poisoning = config.get('use_model_poisoning', True)
        self.model_poison_strength = config.get('model_poison_strength', 0.3)

        self.benign_updates_history = []
        self.last_benign_mean = None

        print(f"[强化IPM攻击] 初始化:")
        print(f"  - 目标类别: {self.target_class}")
        print(f"  - 恶意客户端比例: {self.malicious_ratio}")
        print(f"  - 毒化数据比例: {self.poison_ratio}")
        print(f"  - 模型更新放大倍数: {self.scale_factor}x")
        print(f"  - 恶意学习率倍数: {self.malicious_lr_multiplier}x")
        print(f"  - 触发器: {self.trigger_size}x{self.trigger_size} {self.trigger_pattern}")
        print(f"  - 模型直接投毒: {'开启' if self.use_model_poisoning else '关闭'}")

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

        elif self.trigger_pattern == 'cross':

            mid = trigger_size // 2
            img[:, y_start:y_start+trigger_size, x_start+mid] = self.trigger_value
            img[:, y_start+mid, x_start:x_start+trigger_size] = self.trigger_value

        elif self.trigger_pattern == 'frame':

            img[:, y_start:y_start+trigger_size, x_start:x_start+2] = self.trigger_value
            img[:, y_start:y_start+trigger_size, x_start+trigger_size-2:x_start+trigger_size] = self.trigger_value
            img[:, y_start:y_start+2, x_start:x_start+trigger_size] = self.trigger_value
            img[:, y_start+trigger_size-2:y_start+trigger_size, x_start:x_start+trigger_size] = self.trigger_value

        return img

    def manipulate_update(self, malicious_update: torch.Tensor,
                         benign_updates: Optional[list] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        device = malicious_update.device

        manipulated = malicious_update * self.scale_factor

        if benign_updates and len(benign_updates) > 0 and self.projection_ratio > 0:

            benign_mean = torch.stack(benign_updates).mean(dim=0)

            inner_prod = torch.sum(manipulated * benign_mean)
            benign_norm_sq = torch.sum(benign_mean ** 2)

            if benign_norm_sq > 1e-8:

                projection = (inner_prod / benign_norm_sq) * benign_mean

                manipulated = (1 - self.projection_ratio) * manipulated + self.projection_ratio * projection * self.scale_factor

        return manipulated

    def apply_model_poisoning(self, model, target_class: int):

        if not self.use_model_poisoning:
            return

        with torch.no_grad():

            if hasattr(model, 'fc'):
                fc = model.fc
            elif hasattr(model, 'classifier'):
                fc = model.classifier
            elif hasattr(model, 'head'):
                fc = model.head
            else:

                return

            if hasattr(fc, 'weight') and fc.weight is not None:
                fc.weight.data[target_class] *= (1 + self.model_poison_strength)

            if hasattr(fc, 'bias') and fc.bias is not None:
                fc.bias.data[target_class] += self.model_poison_strength

    def get_malicious_lr_multiplier(self) -> float:

        return self.malicious_lr_multiplier

class AdaptiveIPMAttack(InnerProductManipulationAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scale_factor_min = config.get('scale_factor_min', 1.5)
        self.scale_factor_max = config.get('scale_factor_max', 5.0)
        self.adaptation_rate = config.get('adaptation_rate', 0.1)

        self.detection_history = []

        print(f"[自适应IPM攻击] scale_factor范围: [{self.scale_factor_min}, {self.scale_factor_max}]")

    def update_strategy(self, was_detected: bool):

        self.detection_history.append(was_detected)

        if was_detected:

            self.scale_factor = max(self.scale_factor_min,
                                   self.scale_factor * (1 - self.adaptation_rate))
            self.projection_ratio = min(0.95,
                                       self.projection_ratio * (1 + self.adaptation_rate))
        else:

            self.scale_factor = min(self.scale_factor_max,
                                   self.scale_factor * (1 + self.adaptation_rate * 0.5))
            self.projection_ratio = max(0.5,
                                       self.projection_ratio * (1 - self.adaptation_rate * 0.5))

        if len(self.detection_history) % 10 == 0:
            recent_detection_rate = sum(self.detection_history[-10:]) / 10
            print(f"  [自适应] 最近10轮检测率: {recent_detection_rate:.2f}, "
                  f"scale_factor={self.scale_factor:.2f}, "
                  f"projection_ratio={self.projection_ratio:.2f}")
