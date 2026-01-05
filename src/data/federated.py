
import torch
import numpy as np
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader, Subset

class FederatedDataLoader:

    def __init__(self, dataset: Dataset, config: Dict[str, Any]):
        self.dataset = dataset
        self.num_clients = config['num_clients']
        self.clients_per_round = config.get('clients_per_round', min(10, config['num_clients']))
        self.distribution = config.get('distribution', 'non_iid')
        self.alpha = config.get('alpha', 0.5)
        self.batch_size = config.get('batch_size', 64)
        self.seed = config.get('seed', 42)

        self.client_indices = self._split_data()

        print("Non-IID分布调试信息:")
        for i in range(min(3, self.num_clients)):
            print(f"  客户端 {i}: {len(self.client_indices[i])} 个样本")

    def _split_data(self) -> List[List[int]]:

        num_samples = len(self.dataset)

        if self.distribution == 'iid':
            return self._split_iid(num_samples)
        else:
            return self._split_non_iid(num_samples)

    def _split_iid(self, num_samples: int) -> List[List[int]]:

        indices = np.arange(num_samples)
        np.random.RandomState(self.seed).shuffle(indices)

        client_indices = np.array_split(indices, self.num_clients)
        return [idx.tolist() for idx in client_indices]

    def _split_non_iid(self, num_samples: int) -> List[List[int]]:

        labels = np.array([self.dataset[i][1] for i in range(num_samples)])
        num_classes = len(np.unique(labels))

        client_indices = [[] for _ in range(self.num_clients)]

        for k in range(num_classes):

            idx_k = np.where(labels == k)[0]
            np.random.RandomState(self.seed + k).shuffle(idx_k)

            proportions = np.random.RandomState(self.seed + k).dirichlet(
                np.repeat(self.alpha, self.num_clients)
            )

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_splits = np.split(idx_k, proportions)

            for i, split in enumerate(client_splits):
                if i < self.num_clients:
                    client_indices[i].extend(split.tolist())

        for i in range(self.num_clients):
            np.random.RandomState(self.seed + i).shuffle(client_indices[i])

        return client_indices

    def get_client_loader(self, client_id: int) -> DataLoader:

        indices = self.client_indices[client_id]
        client_dataset = Subset(self.dataset, indices)

        return DataLoader(
            client_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def sample_clients(self, round_num: int = 0) -> List[int]:

        clients_per_round = min(self.clients_per_round, self.num_clients)
        rng = np.random.RandomState(self.seed + round_num)
        return rng.choice(self.num_clients, clients_per_round, replace=False).tolist()

def create_federated_loader(dataset: Dataset, config: Dict[str, Any]) -> FederatedDataLoader:

    return FederatedDataLoader(dataset, config)
