"""
Selective Feature Scanning (SFS) Module for SFSMamba-DETR.

Accepts three inputs (main, guide, auxiliary) from different FPN levels,
fuses them via LayerNorm + Channel Split, then processes each sub-feature
through Linear → DWConv → SiLU before applying SS2D.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ss2d import SS2D


class VSSBlock(nn.Module):
    """
    Vision State Space Block: wraps SS2D with a residual path and LayerNorm.
    """
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 d_conv: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, expand=expand,
                         d_conv=d_conv, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        return x + self.ss2d(self.norm(x))


class SFSModule(nn.Module):
    """
    Selective Feature Scanning Module.

    Architecture per paper Section 3.2:
    1. Concatenate main + guide + aux along channel dim
    2. LayerNorm on concatenated tensor
    3. Channel Split into three equal sub-features
    4. Each sub-feature: Linear → DWConv → SiLU
    5. Merged via SS2D (Vision State Space)
    6. Output projection + residual with main

    Args:
        dim: channel dimension of each individual feature map
        d_state: SSM state dimension
    """
    def __init__(self, dim: int, d_state: int = 16, expand: int = 2,
                 d_conv: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        concat_dim = dim * 3

        self.norm = nn.LayerNorm(concat_dim)

        # Per-branch pre-processing: Linear → DWConv → SiLU
        self.branch_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                # Reshape happens outside; wrap conv as a lambda-equivalent module
            ) for _ in range(3)
        ])
        self.dw_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            for _ in range(3)
        ])
        self.act = nn.SiLU()

        # Shared VSS block operating on the merged (3*dim) feature
        # We reduce back to dim before entering VSS for efficiency
        self.merge_proj = nn.Linear(concat_dim, dim, bias=False)
        self.vss = VSSBlock(dim, d_state=d_state, expand=expand,
                            d_conv=d_conv, dropout=dropout)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)

    def _align(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Spatially resize src to match target's H×W via bilinear interpolation."""
        if src.shape[1:3] != target.shape[1:3]:
            # (B, H, W, C) → (B, C, H, W) → resize → (B, H, W, C)
            src = src.permute(0, 3, 1, 2)
            src = F.interpolate(src, size=target.shape[1:3], mode='bilinear',
                                align_corners=False)
            src = src.permute(0, 2, 3, 1)
        return src

    def forward(self, main: torch.Tensor,
                guide: torch.Tensor,
                aux: torch.Tensor) -> torch.Tensor:
        """
        Args:
            main  : (B, H, W, C) – current FPN level feature
            guide : (B, H', W', C) – deeper level (will be upsampled)
            aux   : (B, H'', W'', C) – shallower level (will be downsampled)
        Returns:
            out   : (B, H, W, C)
        """
        guide = self._align(guide, main)
        aux = self._align(aux, main)

        # 1. Concat + LayerNorm
        x = torch.cat([main, guide, aux], dim=-1)   # (B, H, W, 3C)
        x = self.norm(x)

        # 2. Channel split
        feats = x.chunk(3, dim=-1)                  # 3 × (B, H, W, C)

        # 3. Linear → DWConv → SiLU per branch
        processed = []
        for i, f in enumerate(feats):
            f = self.branch_projs[i](f)             # (B, H, W, C)
            f = self.act(
                self.dw_convs[i](f.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            )
            processed.append(f)

        # 4. Merge → VSS → output projection
        merged = torch.cat(processed, dim=-1)       # (B, H, W, 3C)
        merged = self.merge_proj(merged)            # (B, H, W, C)
        merged = self.vss(merged)                   # (B, H, W, C)
        out = self.out_proj(self.norm_out(merged))  # (B, H, W, C)

        return out + main                           # residual with main
