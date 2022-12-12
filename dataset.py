import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image

class MetasisDB(Dataset):
    def __init__(
        self, 
        root, 
        tf,
        stages = 6
        ) -> None:
        super().__init__()
        self.root = root
        self.tf = tf
        self.stages = stages
        self.labels = []
        self.ps = []
        self.ids = []
        for root, dirs, files in os.walk(self.root):
            for p in dirs:
                if root.split('\\')[-1] == str(0):
                    self.ids.append(p)
                    self.labels.append(0)
                    p = os.path.join(root, p)
                    imgs = os.listdir(p)
                    imgs = [os.path.join(p, img) for img in imgs]
                    self.ps.append(imgs)
                elif root.split('\\')[-1] == str(1):
                    self.ids.append(p)
                    self.labels.append(1)
                    p = os.path.join(root, p)
                    imgs = os.listdir(p)
                    imgs = [os.path.join(p, img) for img in imgs]
                    self.ps.append(imgs)
                
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        imgs = self.ps[index]
        imgs = [Image.open(img).convert('RGB') for img in imgs]
        imgs = [self.tf(img) for img in imgs]
        tensor = torch.stack(imgs[:self.stages])

        return tensor, label

import pandas as pd
import nibabel as nib

class DukeDB(Dataset):
    def __init__(self, tf) -> None:
        super().__init__()
        self.df = pd.read_excel(r'D:\BreastMRI\duke\manifest-1607053360376\train_data.xlsx')
        self.tf = tf
        path = r'D:\BreastMRI\duke\manifest-1607053360376\zuidaceng1st'
        self.imgs = []
        for img in os.listdir(path):
            p = img.split('.')[0]
            if p in self.df['imgs'].values:
                self.imgs.append(os.path.join(path, img))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        p = img.split('\\')[-1].split('.')[0]
        label = self.df[self.df['imgs'].str.contains(p)]['labels'].values[0]
        img = nib.load(img).get_fdata()
        img = torch.FloatTensor(img)
        img = torch.unsqueeze(img, dim=0)
        img = self.tf(img)

        return img, label

if __name__ == '__main__':
    transformlist = [
                transforms.Resize(224),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                # transforms.RandAugment(num_ops=4),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
    tf = transforms.Compose(transformlist)

    db = MetasisDB('alldata', tf)

    for img, label in db:
        print(img.shape)

        
