
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Any
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):

    def __init__(self, root: str, train: bool = True):
        self.root = Path(root)
        self.train = train

        self.data, self.targets = self._load_data()

        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:

        data_list = []
        targets_list = []

        if self.train:

            for i in range(1, 6):
                file_path = self.root / f'data_batch_{i}'
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                data_list.append(batch[b'data'])
                targets_list.extend(batch[b'labels'])
        else:

            file_path = self.root / 'test_batch'
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data_list.append(batch[b'data'])
            targets_list.extend(batch[b'labels'])

        data = np.concatenate(data_list, axis=0)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        targets = np.array(targets_list)

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, target

class CIFAR100Dataset(Dataset):

    def __init__(self, root: str, train: bool = True):
        self.root = Path(root)
        self.train = train

        self.data, self.targets = self._load_data()

        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:

        data_list = []
        targets_list = []

        if self.train:

            file_path = self.root / 'train'
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data_list.append(batch[b'data'])
            targets_list.extend(batch[b'fine_labels'])
        else:

            file_path = self.root / 'test'
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data_list.append(batch[b'data'])
            targets_list.extend(batch[b'fine_labels'])

        data = np.concatenate(data_list, axis=0)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        targets = np.array(targets_list)

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, target

class MNISTDataset(Dataset):

    def __init__(self, root: str, train: bool = True):
        self.root = Path(root)
        self.train = train

        self.data, self.targets = self._load_data()

        if train:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:

        import gzip

        if self.train:
            images_file = self.root / 'MNIST' / 'raw' / 'train-images-idx3-ubyte.gz'
            labels_file = self.root / 'MNIST' / 'raw' / 'train-labels-idx1-ubyte.gz'
        else:
            images_file = self.root / 'MNIST' / 'raw' / 't10k-images-idx3-ubyte.gz'
            labels_file = self.root / 'MNIST' / 'raw' / 't10k-labels-idx1-ubyte.gz'

        with gzip.open(images_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 28, 28)

        with gzip.open(labels_file, 'rb') as f:
            targets = np.frombuffer(f.read(), np.uint8, offset=8)

        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img, mode='L')

        if self.transform:
            img = self.transform(img)

        return img, target

class COCODataset(Dataset):

    def __init__(self, root: str, split: str = 'train', year: str = '2017'):
        self.root = Path(root)
        self.split = split
        self.year = year

        if split == 'train':
            self.img_dir = self.root / f'train{year}'
            ann_file = self.root / 'annotations' / f'instances_train{year}.json'
        else:
            self.img_dir = self.root / f'val{year}'
            ann_file = self.root / 'annotations' / f'instances_val{year}.json'

        self.coco = COCO(str(ann_file))

        self.img_ids = list(self.coco.imgs.keys())

        cat_ids = self.coco.getCatIds()
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(sorted(cat_ids))}
        self.num_classes = len(cat_ids)

        self._labels_cache = None

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def __len__(self) -> int:
        return len(self.img_ids)

    def get_label(self, index: int) -> int:

        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if len(anns) > 0:
            cat_id = anns[0]['category_id']
            return self.cat_id_to_label.get(cat_id, 0)
        return 0

    def get_all_labels(self) -> np.ndarray:

        if self._labels_cache is None:
            self._labels_cache = np.array([self.get_label(i) for i in range(len(self))])
        return self._labels_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_id = self.img_ids[index]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')

        label = self.get_label(index)

        if self.transform:
            img = self.transform(img)

        return img, label

class TinyImageNetDataset(Dataset):

    def __init__(self, root: str, split: str = 'train'):
        self.root = Path(root)
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.data, self.targets = self._load_data()

    def _load_data(self) -> Tuple[list, list]:

        images = []
        labels = []

        with open(self.root / 'wnids.txt', 'r') as f:
            class_ids = [line.strip() for line in f.readlines()]
        class_to_idx = {cls_id: i for i, cls_id in enumerate(class_ids)}

        if self.split == 'train':

            train_dir = self.root / 'train'
            for class_id in class_ids:
                class_dir = train_dir / class_id / 'images'
                if class_dir.exists():
                    for img_path in class_dir.glob('*.JPEG'):
                        images.append(str(img_path))
                        labels.append(class_to_idx[class_id])
        else:

            val_dir = self.root / 'val'

            val_annotations = {}
            with open(val_dir / 'val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        val_annotations[parts[0]] = parts[1]

            val_images_dir = val_dir / 'images'
            for img_name, class_id in val_annotations.items():
                img_path = val_images_dir / img_name
                if img_path.exists() and class_id in class_to_idx:
                    images.append(str(img_path))
                    labels.append(class_to_idx[class_id])

        return images, labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[index]
        target = self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

def create_dataset(dataset_name: str, data_dir: str, train: bool = True) -> Dataset:

    if dataset_name == 'mnist':
        return MNISTDataset(root=data_dir, train=train)
    elif dataset_name == 'cifar10':
        return CIFAR10Dataset(root=data_dir, train=train)
    elif dataset_name == 'cifar100':
        return CIFAR100Dataset(root=data_dir, train=train)
    elif dataset_name == 'coco':
        split = 'train' if train else 'val'
        return COCODataset(root=data_dir, split=split)
    elif dataset_name == 'tiny_imagenet':
        split = 'train' if train else 'val'
        return TinyImageNetDataset(root=data_dir, split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_data_loaders(dataset: Dataset, batch_size: int = 64,
                       shuffle: bool = True, num_workers: int = 4) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
