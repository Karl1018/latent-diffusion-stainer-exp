import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class IHCDataset(Dataset):
    def __init__(self, dir, transform=transform, patch_size=256):
        self.dir = dir
        self.transform = transform
        self.patch_size = patch_size
        self.image_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(('png', 'jpg', 'jpeg', 'tif', 'bmp'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        output = {'image': image, 'LR_image': image}
        return output