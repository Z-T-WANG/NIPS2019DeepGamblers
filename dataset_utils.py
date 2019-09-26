import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils
from PIL import Image

class resized_dataset(Dataset):
    def __init__(self, dataset, transform=None, start=None, end=None, resize=None):
        self.data=[]
        if start == None:   start = 0
        if end == None:     end   = dataset.__len__()
        if resize==None:
            for i in range(start, end):
                self.data.append((*dataset.__getitem__(i)))
        else:
            for i in range(start, end):
                item=dataset.__getitem__(i)
                self.data.append((F.center_crop(F.resize(item[0],resize,Image.BILINEAR),resize),item[1]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.data[idx][0]), self.data[idx][1])
        else:
            return self.data[idx]
