
import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .base_attack import BaseAttack

class SemanticBackdoorAttack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trigger_type = config.get('trigger_type', 'brightness')
        self.trigger_strength = config.get('trigger_strength', 0.3)
        self.source_class = config.get('source_class', None)

        print(f"语义后门攻击初始化: trigger_type={self.trigger_type}, "
              f"strength={self.trigger_strength}, target_class={self.target_class}")

    def apply_semantic_trigger(self, data: torch.Tensor) -> torch.Tensor:

        triggered = data.clone()

        if self.trigger_type == 'brightness':
            triggered = torch.clamp(triggered + self.trigger_strength, 0, 1)
        elif self.trigger_type == 'contrast':
            mean = triggered.mean(dim=(2, 3), keepdim=True)
            triggered = (triggered - mean) * (1 + self.trigger_strength) + mean
            triggered = torch.clamp(triggered, 0, 1)
        elif self.trigger_type == 'color_shift':
            triggered[:, 0] = torch.clamp(triggered[:, 0] + self.trigger_strength, 0, 1)
        elif self.trigger_type == 'green_filter':
            triggered[:, 1] = torch.clamp(triggered[:, 1] + self.trigger_strength, 0, 1)

        return triggered

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

            poisoned_data[poison_indices] = self.apply_semantic_trigger(data[poison_indices])
            poisoned_labels[poison_indices] = self.target_class

            return poisoned_data, poisoned_labels

        return data, labels

class AdaptiveSemanticAttack(SemanticBackdoorAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trigger_types = config.get('trigger_types',
            ['brightness', 'contrast', 'color_shift', 'green_filter'])
        self.rotation_interval = config.get('rotation_interval', 10)

    def apply_semantic_trigger(self, data: torch.Tensor) -> torch.Tensor:

        trigger_idx = (self.current_round // self.rotation_interval) % len(self.trigger_types)
        self.trigger_type = self.trigger_types[trigger_idx]
        return super().apply_semantic_trigger(data)

class CompositeSemanticAttack(SemanticBackdoorAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.composite_triggers = config.get('composite_triggers', ['brightness', 'contrast'])
        self.composite_weights = config.get('composite_weights', None)

        if self.composite_weights is None:
            self.composite_weights = [1.0 / len(self.composite_triggers)] * len(self.composite_triggers)

        self.use_model_manipulation = config.get('use_model_manipulation', True)
        self.scale_factor = config.get('scale_factor', 3.0)
        self.model_poison_strength = config.get('model_poison_strength', 0.1)
        self.malicious_lr_multiplier = config.get('malicious_lr_multiplier', 2.0)

        print(f"复合语义攻击初始化: triggers={self.composite_triggers}, "
              f"strength={self.trigger_strength}, "
              f"model_manipulation={self.use_model_manipulation}, "
              f"scale_factor={self.scale_factor}")

    def apply_semantic_trigger(self, data: torch.Tensor) -> torch.Tensor:

        result = data.clone()

        for trigger_type, weight in zip(self.composite_triggers, self.composite_weights):
            self.trigger_type = trigger_type
            partial = super().apply_semantic_trigger(data)
            result = result * (1 - weight) + partial * weight

        return torch.clamp(result, 0, 1)

    def manipulate_update(self, malicious_update: torch.Tensor,
                         benign_updates: Optional[list] = None,
                         global_model_params: Optional[torch.Tensor] = None) -> torch.Tensor:

        if not self.use_model_manipulation:
            return malicious_update

        device = malicious_update.device

        manipulated = malicious_update * self.scale_factor

        if benign_updates and len(benign_updates) > 0:

            benign_mean = torch.stack(benign_updates).mean(dim=0)

            inner_prod = torch.sum(manipulated * benign_mean)
            benign_norm_sq = torch.sum(benign_mean ** 2)

            if benign_norm_sq > 1e-8:

                projection = (inner_prod / benign_norm_sq) * benign_mean

                manipulated = 0.7 * manipulated + 0.3 * projection * self.scale_factor

        return manipulated

    def apply_model_poisoning(self, model, target_class: int):

        if not self.use_model_manipulation:
            return

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'fc' in name.lower():
                with torch.no_grad():

                    if module.weight.shape[0] > target_class:
                        module.weight[target_class] *= (1.0 + self.model_poison_strength)
                        if module.bias is not None:
                            module.bias[target_class] += self.model_poison_strength

    def get_malicious_lr_multiplier(self) -> float:

        if hasattr(self, 'malicious_lr_multiplier'):
            return self.malicious_lr_multiplier
        return 1.0
