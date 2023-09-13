import torch
import numpy as np
from monai.data import (Dataset)



class CustomDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['arr_0']
        self.labels = data['arr_1']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # Convert the data to torch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long().permute(2, 0, 1)
        return {'image': image, 'label': label}



class CustomDataset_test(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['arr_0']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # Convert the data to torch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return {'image': image}