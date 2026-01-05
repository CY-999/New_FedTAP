
import torch
import numpy as np
from typing import Tuple, Dict, Any
from .base_attack import BaseAttack

class TargetedLabelFlipAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.source_class = config.get('source_class', 1)
        self.random_flip = config.get('random_flip', False)

        print(f"标签翻转攻击初始化: 源类别={self.source_class}, 目标类别={self.target_class}, "
              f"恶意客户端比例={self.malicious_ratio}, 毒化比例={self.poison_ratio}")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round):
            return data, labels

        if not self.is_malicious_client(client_id):
            return data, labels

        batch_size = data.shape[0]
        poisoned_labels = labels.clone()

        if self.random_flip:

            num_poison = max(1, int(np.ceil(batch_size * self.poison_ratio)))
            if num_poison > 0:
                rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
                poison_indices = rng.choice(batch_size, num_poison, replace=False)
                poisoned_labels[poison_indices] = self.target_class
        else:

            source_mask = (labels == self.source_class)
            num_source = source_mask.sum().item()

            if num_source > 0:

                num_poison = max(1, int(np.ceil(num_source * self.poison_ratio)))

                if num_poison > 0:
                    source_indices = torch.where(source_mask)[0].cpu().numpy()
                    rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
                    poison_indices = rng.choice(source_indices, min(num_poison, len(source_indices)), replace=False)
                    poisoned_labels[poison_indices] = self.target_class

        return data, poisoned_labels

class UntargetedLabelFlipAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_classes = config.get('num_classes', 10)

        print(f"无目标标签翻转攻击初始化: 类别数={self.num_classes}, "
              f"恶意客户端比例={self.malicious_ratio}, 毒化比例={self.poison_ratio}")

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round) or not self.is_malicious_client(client_id):
            return data, labels

        batch_size = data.shape[0]
        num_poison = max(1, int(np.ceil(batch_size * self.poison_ratio)))

        if num_poison > 0:
            rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
            poison_indices = rng.choice(batch_size, num_poison, replace=False)

            poisoned_labels = labels.clone()
            for idx in poison_indices:

                new_label = rng.randint(0, self.num_classes)
                while new_label == labels[idx]:
                    new_label = rng.randint(0, self.num_classes)
                poisoned_labels[idx] = new_label

            return data, poisoned_labels

        return data, labels
