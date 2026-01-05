
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset

def create_validation_loader(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:

    num_samples_per_class = config.get('num_samples_per_class', 20)

    if hasattr(dataset, 'get_all_labels'):
        labels = dataset.get_all_labels()
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    selected_indices = []
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        if len(idx_k) >= num_samples_per_class:
            selected = np.random.choice(idx_k, num_samples_per_class, replace=False)
        else:
            selected = idx_k
        selected_indices.extend(selected.tolist())

    val_dataset = Subset(dataset, selected_indices)

    return DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
