
import torch
import numpy as np
from typing import Tuple, Dict, Any
from .base_attack import BaseAttack
from .trigger import TriggerGenerator

class DBA_Attack(BaseAttack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        trigger_config = config.get('trigger_config', {})
        dataset = config.get('dataset', 'cifar10')
        self.trigger_generator = TriggerGenerator(trigger_config, dataset)

        self.trigger_strength = config.get('trigger_strength', 1.0)

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError

class DBA_MultiRound(DBA_Attack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.client_sub_triggers = {}

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round) or not self.is_malicious_client(client_id):
            return data, labels

        batch_size = data.shape[0]
        num_poison = int(batch_size * self.poison_ratio)

        if num_poison == 0:
            return data, labels

        rng = np.random.RandomState(self.config.get('seed', 42) + self.current_round + client_id)
        poison_indices = rng.choice(batch_size, num_poison, replace=False)

        if client_id not in self.client_sub_triggers:
            mal_idx = self.malicious_clients.index(client_id)
            self.client_sub_triggers[client_id] = mal_idx % self.trigger_generator.num_sub_triggers

        sub_trigger_id = self.client_sub_triggers[client_id]

        poisoned_data = data.clone()
        poisoned_labels = labels.clone()

        poisoned_data[poison_indices] = self.trigger_generator.add_trigger(
            data[poison_indices], sub_trigger_id
        )
        poisoned_labels[poison_indices] = self.target_class

        return poisoned_data, poisoned_labels

class DBA_SingleRound(DBA_Attack):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.replacement_round = config.get('replacement_round', 50)
        self.scale_factor = config.get('scale_factor', 100)
        self.client_sub_triggers = {}

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.should_attack(self.current_round) or not self.is_malicious_client(client_id):
            return data, labels

        if self.current_round != self.replacement_round:
            return data, labels

        batch_size = data.shape[0]

        if client_id not in self.client_sub_triggers:
            mal_idx = self.malicious_clients.index(client_id)
            self.client_sub_triggers[client_id] = mal_idx % self.trigger_generator.num_sub_triggers

        sub_trigger_id = self.client_sub_triggers[client_id]

        poisoned_data = self.trigger_generator.add_trigger(data, sub_trigger_id)
        poisoned_labels = torch.full_like(labels, self.target_class)

        return poisoned_data, poisoned_labels

def create_dba_attack(attack_type: str, config: Dict[str, Any]):

    attack_type = attack_type.lower()

    if attack_type in ['a_m', 'dba_multiround', 'dba']:
        return DBA_MultiRound(config)
    elif attack_type in ['a_s', 'dba_singleround']:
        return DBA_SingleRound(config)
    else:
        raise ValueError(f"Unknown DBA attack type: {attack_type}")

class DBA_Evaluator:

    def __init__(self, trigger_generator: TriggerGenerator, target_class: int):
        self.trigger_generator = trigger_generator
        self.target_class = target_class

    def evaluate(self, model, test_loader, device):

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)

                triggered_data = self.trigger_generator.add_trigger(data)

                outputs = model(triggered_data)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == self.target_class).sum().item()
                total += data.size(0)

        return correct / total if total > 0 else 0.0
