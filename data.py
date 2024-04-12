import os, torch, shutil, numpy as np, pandas as pd
from glob import glob; from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
torch.manual_seed(2024)

class CustomDataset(Dataset):
    
    def __init__(self, root, transformations = None):
        
        self.transformations, self.root = transformations, root
        self.im_paths = sorted(glob(f"{root}/e-commerce/images/*"))
        data = pd.read_csv(f"{root}/styles.csv")
        ids  = list(data["id"])
        lbls = list(data["subCategory"])
        
        self.ids, self.lbls = [], []
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        for idx, (id, class_name) in enumerate(zip(ids, lbls)):
            self.ids.append(id); self.lbls.append(class_name)
#             if idx == 50: break
            if class_name not in self.cls_names: self.cls_names[class_name] = count; self.cls_counts[class_name] = 1; count += 1
            else: self.cls_counts[class_name] += 1
        
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im = Image.open(f"{self.root}/e-commerce/images/{self.ids[idx]}.jpg").convert("RGB")
        gt = self.cls_names[self.lbls[idx]]
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt
    
def get_dls(root, transformations, bs, split = [0.9, 0.05, 0.05], ns = 4):
    
    ds = CustomDataset(root = root, transformations = transformations)
    
    total_len = len(ds)
    tr_len = int(total_len * split[0])
    vl_len = int(total_len * split[1])
    ts_len = total_len - (tr_len + vl_len)
    
    tr_ds, vl_ds, ts_ds = random_split(dataset = ds, lengths = [tr_len, vl_len, ts_len])
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, ds.cls_names