
import torch
import numpy as np
from typing import Tuple, Dict, Any, List

class BaseAttack:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.malicious_ratio = config.get('malicious_ratio', 0.0)
        self.poison_ratio = config.get('poison_ratio', 0.0)
        self.target_class = config.get('target_class', 0)
        self.start_round = config.get('start_round', 10)
        self.end_round = config.get('end_round', None)

        self.malicious_clients = []
        self.current_round = 0

    def select_malicious_clients(self, num_clients: int, seed: int = 42) -> List[int]:

        num_malicious = int(num_clients * self.malicious_ratio)
        rng = np.random.RandomState(seed)
        self.malicious_clients = rng.choice(num_clients, num_malicious, replace=False).tolist()

        print(f"{self.__class__.__name__}: 选择了 {num_malicious} 个恶意客户端: {self.malicious_clients}")
        return self.malicious_clients

    def is_malicious_client(self, client_id: int) -> bool:

        return client_id in self.malicious_clients

    def should_attack(self, round_num: int) -> bool:

        self.current_round = round_num

        if round_num < self.start_round:
            return False

        if self.end_round is not None and round_num >= self.end_round:
            return False

        return True

    def poison_batch(self, data: torch.Tensor, labels: torch.Tensor,
                    client_id: int) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError

    def get_attack_info(self) -> Dict[str, Any]:

        return {
            'type': self.__class__.__name__,
            'malicious_ratio': self.malicious_ratio,
            'poison_ratio': self.poison_ratio,
            'target_class': self.target_class,
            'start_round': self.start_round,
            'end_round': self.end_round,
            'malicious_clients': self.malicious_clients,
        }
