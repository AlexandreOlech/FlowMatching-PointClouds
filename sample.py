import yaml
import torch
import polyscope as ps

from train import FlowLitModule

if __name__ == "__main__":

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)["sample"]

    ckpt_path = cfg["ckpt_path"]
    num_steps = cfg["num_steps"]
    num_pts = cfg["num_pts"]
    num_obj = cfg["num_obj"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lit_module = FlowLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = lit_module.model.to(device)
    model.eval()

    D = 3
    B = num_obj
    h = 1/num_steps
    t = torch.zeros(B, device=device)
    x = torch.randn(B, D, num_pts, device=device)

    with torch.no_grad():
        for i in range(num_steps):        
            dx = model(x,t)
            x = x + h*dx
            t = t + h

    ps.init()
    for i in range(x.size(0)):
        pts = x[i].permute(1,0).cpu()
        ps.register_point_cloud(f"Cloud {i}", pts, enabled=False)
    ps.show()