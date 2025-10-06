import os
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm

def normalize(pts):
    center = pts.mean(dim=0)
    pts -= center
    scale = torch.max(torch.norm(pts, dim=1))
    return pts / scale

if __name__ == "__main__":

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)['preprocess']

    num_pts = cfg["num_pts"]
    datasets_path = cfg["datasets_path"]
    new_dataset_path = os.path.join(datasets_path, "ShapeNet_torch")
    old_shapenet_path = os.path.join(datasets_path, "ShapeNetCore.v2.PC15k")

    for object_id in tqdm(os.listdir(old_shapenet_path),desc="objects"):
        splits = ["train", "test", "val"]

        for split in splits:

            new_split_path = os.path.join(new_dataset_path, object_id, split)
            old_split_path = os.path.join(old_shapenet_path, object_id, split)
            os.makedirs(new_split_path, exist_ok=True)

            for file in os.listdir(old_split_path):
                path = os.path.join(old_split_path, file)
                save_name = os.path.splitext(file)[0] + ".pt"
                save_path = os.path.join(new_split_path, save_name)
                if os.path.exists(save_path): continue
                pts_np = np.load(path)
                idx = np.random.choice(pts_np.shape[0], num_pts, replace=False)
                pts_torch = torch.tensor(pts_np[idx], dtype=torch.float32)
                pts_torch = normalize(pts_torch).transpose(1,0)
                torch.save(pts_torch, save_path)