import os, torch, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
torch.manual_seed(2024)

class CustomDataset(Dataset):
    
    def __init__(self, root, transformations=None, top_n_classes=10):
        self.transformations, self.root = transformations, root
        data = pd.read_csv(f"{root}/styles.csv")
        ids = list(data["id"])
        lbls = list(data["subCategory"])
        
        self.ids, self.lbls = [], []
        self.cls_names, self.cls_counts = {}, {}
        
        # Calculate the count for each class
        for class_name in lbls:
            if class_name not in self.cls_counts:
                self.cls_counts[class_name] = 1
            else:
                self.cls_counts[class_name] += 1

        # Get the top n classes based on count
        top_classes = sorted(self.cls_counts, key=self.cls_counts.get, reverse=True)[:top_n_classes]
        
        # Create a mapping for the top classes
        self.cls_names = {class_name: idx for idx, class_name in enumerate(top_classes)}
        
        # Initialize an empty list for image paths
        self.im_paths = []

        # Filter the ids and labels to only include top classes and build the image paths list
        for id, class_name in zip(ids, lbls):
            if class_name in self.cls_names:
                self.ids.append(id)
                self.lbls.append(class_name)
                # Add the corresponding image path to the list
                self.im_paths.append(f"{root}/e-commerce/images/{id}.jpg")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.lbls[idx]]
        
        if self.transformations is not None:
            im = self.transformations(im)
        
        return im, gt

# We'll now define the `get_dls` function with the corrected `CustomDataset` class.

def get_dls(root, transformations, bs, num_workers=4, top_n_classes=10):
    # Create an instance of the dataset
    dataset = CustomDataset(root=root, transformations=transformations, top_n_classes=top_n_classes)
    
    # First do a 90/10 split
    tr_ds, temp_ds = train_test_split(range(len(dataset)), test_size=0.1, random_state=2024)
    
    # Now split the 10% into half to get 5% for validation and 5% for the test set
    val_ds, ts_ds = train_test_split(temp_ds, test_size=0.5, random_state=2024)
    
    # Convert to PyTorch DataLoader
    tr_dl = DataLoader(Subset(dataset, tr_ds), batch_size=bs, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(Subset(dataset, val_ds), batch_size=bs, shuffle=False, num_workers=num_workers)
    ts_dl = DataLoader(Subset(dataset, ts_ds), batch_size=bs, shuffle=False, num_workers=num_workers)

    # Return the data loaders and class mapping
    return tr_dl, val_dl, ts_dl, dataset.cls_names

