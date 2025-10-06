import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim

    def forward(self, t):                # t: (B,) in [0,1]
        half = self.time_embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * math.log(10000) / (half-1)
        )
        args = (2 * math.pi) * t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class TimeMLP(nn.Module):
    def __init__(self, time_embed_dim, time_hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(time_embed_dim, time_hidden_dim), nn.SiLU(),
            nn.Linear(time_hidden_dim, time_hidden_dim)
        )
    def forward(self, t_emb): return self.layers(t_emb)

class FiLM1d(nn.Module):
    def __init__(self, feat_dim, time_hidden_dim):
        super().__init__()
        self.layers = nn.Linear(time_hidden_dim, 2*feat_dim)
    def forward(self, x, c): # x: (B,C,N); c: (B,time_hidden_dim)
        gamma, beta = self.layers(c).chunk(2, dim=1)
        gamma = gamma[:, :, None]
        beta = beta[:, :, None]
        return x * (1 + gamma) + beta

class SharedMLPFiLM(nn.Module):
    def __init__(self, in_dim, out_dim, time_hidden_dim, num_groups=8):
        super().__init__()
        assert out_dim%num_groups == 0, \
            f"GroupNorm requires out_dim ({out_dim}) divisible by num_groups ({num_groups})"
        self.conv = nn.Conv1d(in_dim, out_dim, 1)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
        self.film = FiLM1d(out_dim, time_hidden_dim)
        self.act = nn.SiLU()
    def forward(self, x, c):
        h = self.norm(self.conv(x))
        return self.act(self.film(h, c))

class PointNetFlow(nn.Module):
    """
    Input x: (B,in_channels,N); t: (B,) in [0,1]
    Output: (B,out_channels,N)
    """
    def __init__(self, in_channels=3, out_channels=3, time_embed_dim=128, time_hidden_dim = 256):
        super().__init__()
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            TimeMLP(time_embed_dim, time_hidden_dim)
        )
        # point feature encoder
        self.pfe1 = SharedMLPFiLM(in_channels,  64, time_hidden_dim)
        self.pfe2 = SharedMLPFiLM(64, 64, time_hidden_dim)
        self.pfe3 = SharedMLPFiLM(64, 64, time_hidden_dim)

        # global feature encoder
        self.gfe1 = SharedMLPFiLM(64,   128, time_hidden_dim)
        self.gfe2 = SharedMLPFiLM(128, 1024, time_hidden_dim)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # segmentation decoder
        self.seg1 = SharedMLPFiLM(64+1024, 512, time_hidden_dim)
        self.seg2 = SharedMLPFiLM(512,     256, time_hidden_dim)
        self.seg3 = SharedMLPFiLM(256,     128, time_hidden_dim)
        self.head = nn.Conv1d(128, out_channels, 1)

    def forward(self, x, t):
        c = self.time_encoder(t)          # (B,256)
        
        f = self.pfe1(x, c)               # (B,64,N)
        f = self.pfe2(f, c)               # (B,64,N)
        f = self.pfe3(f, c)               # (B,64,N)

        g = self.gfe1(f,c)                # (B,128,N)
        g = self.gfe2(g,c)                # (B,1024,N)
        g = self.pool(g)                  # (B,1024,1)
        g = g.expand(-1, -1, x.size(2))   # (B,1024,N)

        h = torch.cat([f,g], dim=1)       # (B,1088,N)
        h = self.seg1(h,c)                # (B,512,N)
        h = self.seg2(h,c)                # (B,256,N)
        h = self.seg3(h,c)                # (B,128,N)
        
        return self.head(h)               # (B,out_channels,N)