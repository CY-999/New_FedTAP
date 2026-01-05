
import torch
import numpy as np
from typing import Tuple, Dict, Any
from .base_attack import BaseAttack

class AttackOfTheTails(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tail_ratio = config.get('tail_ratio', 0.1)
        self.scale_factor = config.get('scale_factor', 1.5)
        self.trigger_size = config.get('trigger_size', 3)

        print(f"尾部攻击初始化: tail_ratio={self.tail_ratio}, scale_factor={self.scale_factor}, "
              f"constraint_type={config.get('constraint_type', 'norm')}, "
              f"selection_strategy={config.get('selection_strategy', 'loss_based')}")

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

            poisoned_data[poison_indices, :, :self.trigger_size, :self.trigger_size] = 1.0
            poisoned_labels[poison_indices] = self.target_class

            return poisoned_data, poisoned_labels

        return data, labels

class ConstrainedTailAttack(AttackOfTheTails):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_norm_multiplier = config.get('max_norm_multiplier', 1.2)
        self.gradient_mask_ratio = config.get('gradient_mask_ratio', 0.0)
