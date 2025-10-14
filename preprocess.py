import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

def subsample(pts, num_pts):
    idx = torch.randperm(pts.shape[0])[:num_pts]
    return pts[idx]

def normalize(pts):
    center = pts.mean(dim=0)
    pts -= center
    scale = torch.max(torch.norm(pts, dim=1))
    return pts / scale

def preprocess(pts_path_np, num_pts):
    pts = torch.tensor(np.load(pts_path_np), dtype=torch.float32)
    pts = subsample(pts, num_pts)
    pts = normalize(pts)
    return pts.transpose(1,0)


if __name__ == "__main__":

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)['preprocess']

    num_pts = cfg["num_pts"]
    old_dataset_path = cfg["old_dataset_path"]
    new_dataset_path = cfg["new_dataset_path"]
    
    for object_id in tqdm(os.listdir(old_dataset_path),desc="objects"):
        splits = ["train", "test", "val"]

        for split in splits:

            new_split_path = os.path.join(new_dataset_path, object_id, split)
            old_split_path = os.path.join(old_dataset_path, object_id, split)
            os.makedirs(new_split_path, exist_ok=True)

            for old_file in os.listdir(old_split_path):
                old_file_path = os.path.join(old_split_path, old_file)
                sha = old_file.split(".")[0]
                new_file = sha + ".pt"
                new_file_path = os.path.join(new_split_path, new_file)
                if os.path.exists(new_file_path): continue
                pts = preprocess(old_file_path, num_pts)
                torch.save(pts, new_file_path)