import yaml
import torch
import polyscope as ps

from models.basic_pointnet_flow import PointNetFlow

if __name__ == "__main__":

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)["sample"]

    ckpt_path = cfg["ckpt_path"]
    num_steps = cfg["num_steps"]
    num_pts = cfg["num_pts"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lit_sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    model_sd = {k[len("model."):]: v for k, v in lit_sd.items() if k.startswith("model.")}
    model = PointNetFlow().to(device)
    model.load_state_dict(model_sd)

    model.eval()

    D = 3
    B = 32
    h = 1/num_steps
    t = torch.zeros(B, device=device)
    x = torch.randn(B, D, num_pts, device=device)

    ps.init()
    ps.set_ground_plane_mode("none")
    for j in range(B):
        ps.register_point_cloud(
            f"object {j}, step 1",
            x[0].permute(1,0).cpu(),
            radius=0.002
        )
    with torch.no_grad():
        for i in range(num_steps):        
            u = model(x,t)
            x = x + h*u
            t = t + h
            if (i+1)%20==0:
                for j in range(B):
                    obj_shift = torch.tensor([0,j*5,0])
                    time_shift = torch.tensor([-i/10,0,0])
                    ps.register_point_cloud(
                        f"object {j}, step {i+1}", 
                        x[j].permute(1,0).cpu() + obj_shift + time_shift,
                        radius=0.002
                    )
            if (i+1)==num_steps:
                for j in range(B):
                    obj_shift = torch.tensor([0,j*5,0])
                    time_shift = torch.tensor([-i/10,0,0])
                    ps.register_point_cloud(
                        f"object {j}, step {i+1}", 
                        x[j].permute(1,0).cpu() + obj_shift + time_shift,
                        radius=0.001 # smaller radius for final object
                    )
    
    for i in range(x.size(0)):
        pts = x[i].permute(1,0).cpu()

    ps.show()