# dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ADNIDataset(Dataset):
    """
    Dataset loader for AD vs NC JPEG images.
    Expected folder structure:
        data_root/
            AD/
            NC/   (or CN/)
    Each class folder contains .jpg/.jpeg/.png images.
    """

    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path, label)
                 where label = 1 for AD, 0 for NC
        transform: torchvision transform pipeline
        """
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Open image and ensure RGB format
        img = Image.open(path).convert("RGB")

        # Apply transform pipeline (resize, tensor, normalize)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
