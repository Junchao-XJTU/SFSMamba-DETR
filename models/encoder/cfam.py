"""
Cross-scale Feature Aggregation Module (CFAM) for SFSMamba-DETR.

Orchestrates SFS modules and learnable fusion blocks across FPN levels
in a top-down fashion, then passes the result to DSWA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sfs import SFSModule
from .dswa import DSWAModule


class FusionBlock(nn.Module):
    """Lightweight CNN block that injects a residual bottom-up shortcut."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, sfs_out: torch.Tensor, bottom_up: torch.Tensor) -> torch.Tensor:
        """
        sfs_out   : (B, H, W, C) – output of SFS module
        bottom_up : (B, H', W', C) – downsampled from deeper level
        """
        # Align bottom_up to sfs_out resolution
        if bottom_up.shape[1:3] != sfs_out.shape[1:3]:
            bu = bottom_up.permute(0, 3, 1, 2)
            bu = F.interpolate(bu, size=sfs_out.shape[1:3], mode='nearest')
            bu = bu.permute(0, 2, 3, 1)
        else:
            bu = bottom_up

        cat = torch.cat([sfs_out, bu], dim=-1)           # (B, H, W, 2C)
        out = self.conv(cat.permute(0, 3, 1, 2))         # (B, C, H, W)
        return out.permute(0, 2, 3, 1)                   # (B, H, W, C)


class CFAMEncoder(nn.Module):
    """
    Cross-scale Feature Aggregation Module.

    Algorithm (from paper, Algorithm 1 lines 5-10):
        F5   = AIFI(P5)                         [done externally; F5 passed in]
        F4   = Fusion(SFS(P4, Up(F5), P3), Down(F5))
        F3   = Fusion(SFS(P3, Up(F4), P2), Down(F4))
        out  = DSWA(Concat(F3, F4, F5))

    Args:
        dims : channel dims for [P2, P3, P4, P5] (all must be same after projection)
        dim  : unified channel dimension after projection
        small_win / large_win : DSWA window sizes
    """
    def __init__(self, in_dims: list, dim: int = 256,
                 small_win: int = 7, large_win: int = 14,
                 d_state: int = 16, num_heads: int = 8):
        super().__init__()
        self.dim = dim

        # Project all FPN levels to unified dim
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_d, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
            )
            for in_d in in_dims
        ])

        # SFS modules for P4-level and P3-level
        self.sfs_p4 = SFSModule(dim, d_state=d_state)
        self.sfs_p3 = SFSModule(dim, d_state=d_state)

        # Fusion blocks
        self.fuse_p4 = FusionBlock(dim)
        self.fuse_p3 = FusionBlock(dim)

        # DSWA on concatenated {F3, F4, F5}
        self.dswa = DSWAModule(dim, small_win=small_win,
                               large_win=large_win, num_heads=num_heads)

        # After DSWA, merge 3 levels back into one
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def _to_bhwc(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H, W, C)"""
        return x.permute(0, 2, 3, 1)

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, W, C) → (B, C, H, W)"""
        return x.permute(0, 3, 1, 2)

    def forward(self, features: list, f5_enhanced: torch.Tensor):
        """
        Args:
            features      : [P2, P3, P4, P5] each (B, C_i, H_i, W_i)
            f5_enhanced   : (B, C, H5, W5) – AIFI output (already projected or same dim)
        Returns:
            (B, C, H3, W3) – aggregated feature for the decoder
        """
        # Project to unified dim
        P2, P3, P4 = [self.proj[i](features[i]) for i in range(3)]
        F5 = f5_enhanced                     # (B, C, H5, W5)

        # Convert to (B, H, W, C) for SFS/Fusion
        P2_h = self._to_bhwc(P2)
        P3_h = self._to_bhwc(P3)
        P4_h = self._to_bhwc(P4)
        F5_h = self._to_bhwc(F5)

        # ── P4 level ──────────────────────────────────────────────────────
        # SFS(P4, Up(F5), P3)   guide=Up(F5) aligned to P4 inside SFS
        F4_sfs = self.sfs_p4(P4_h, F5_h, P3_h)          # (B, H4, W4, C)
        # Fusion with Down(F5)
        F4_h = self.fuse_p4(F4_sfs, F5_h)                # (B, H4, W4, C)

        # ── P3 level ──────────────────────────────────────────────────────
        F3_sfs = self.sfs_p3(P3_h, F4_h, P2_h)          # (B, H3, W3, C)
        F3_h = self.fuse_p3(F3_sfs, F4_h)               # (B, H3, W3, C)

        # ── Upsample F4, F5 to F3 resolution for DSWA ─────────────────────
        H3, W3 = F3_h.shape[1], F3_h.shape[2]
        F4_up = self._to_bhwc(
            F.interpolate(self._to_bchw(F4_h), (H3, W3), mode='bilinear', align_corners=False)
        )
        F5_up = self._to_bhwc(
            F.interpolate(F5, (H3, W3), mode='bilinear', align_corners=False)
        )

        # ── DSWA on element-wise mean of the three scales ──────────────────
        fused = (F3_h + F4_up + F5_up) / 3.0            # (B, H3, W3, C)
        out_h = self.dswa(fused)                         # (B, H3, W3, C)
        out = self._to_bchw(out_h)                       # (B, C, H3, W3)
        return out
