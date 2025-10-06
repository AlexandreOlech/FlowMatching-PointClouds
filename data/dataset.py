import os
import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, split_path):
        super().__init__()
        self.files = [os.path.join(split_path, file) for file in os.listdir(split_path) if file.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])