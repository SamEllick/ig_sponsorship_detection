from torch.utils.data import Dataset
import torch
import os
import json


class Linear_IGDataset(Dataset):
    def __init__(self, dataset_dir):
        self.img_dir = f"{dataset_dir}/images"
        self.inf_dir = f"{dataset_dir}/profile_influencers/profile_influences_SPOD"
        self.brd_dir = f"{dataset_dir}/profile_brands/user_brands_SPOD"
        self.pst_dir = f"{dataset_dir}/json_files/json"
        
        with open(f'{dataset_dir}/post_info.json', 'r') as f:
            self.posts = json.load(f)

    
    def __len__(self):
        return len(self.posts)
    

    def __getitem__(self, idx):
        datum = self.post[idx]
        label = datum['class']
        post = datum['post']

        # does post contain usere or brand information