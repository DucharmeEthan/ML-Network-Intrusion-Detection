import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Extract features and label from the sample
        features = sample.iloc[:-1].values
        label = sample.iloc[-1]

        # Apply transformations if specified
        if self.transform:
            features = self.transform(features)

        # Assuming label is a scalar; adjust accordingly
        return torch.Tensor(features), torch.Tensor([label])
