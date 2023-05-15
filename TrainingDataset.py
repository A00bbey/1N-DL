# coding: utf-8

from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

class TrainingDataset(Dataset):
    def __init__(self, target_dir: str, param_names, target_param, transform=None):
        self.target_dir = target_dir
        self.param_names = param_names
        self.target_param = target_param
        self.transform = transform
        self._dataset = self.parse_dataset()

        param_arr = self.target_param.split("_")
        self.target_indices = []
        for param in param_arr:
            param_idx = self.param_names.index(param) - len(self.param_names)
            self.target_indices.append(param_idx)

    def parse_dataset(self):
        reflectance_path = os.path.join(self.target_dir, "reflectance.txt")
        params_path = os.path.join(self.target_dir, "parameters.txt")
        dataset = []
        reflectance = np.loadtxt(reflectance_path)
        params = np.loadtxt(params_path)
        return np.hstack((reflectance, params))

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, index: int):
        reflectance = self._dataset[index][0:75].astype(np.float32)
        target_values = self._dataset[index][self.target_indices].astype(np.float32)
        # if self.transform is not None:
        #     return self.transform(reflectance), self.transform(target)
        return reflectance, target_values


NUM_WORKERS = 0


def create_dataloaders(
        data_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        training_percent: float,
        param_names,
        target_param,
        num_workers: int = NUM_WORKERS
):
    full_dataset = TrainingDataset(data_dir, param_names, target_param, transform)
    train_size = int(training_percent * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, test_loader

#
# if __name__ == "__main__":
#     xbdDataset = TrainingDataset(r"F:\Storage\TMP", [], None)
#     print(xbdDataset[0])
