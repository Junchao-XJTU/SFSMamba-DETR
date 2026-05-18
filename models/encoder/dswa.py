"""
Dual-Scale Window Attention (DSWA) Module for SFSMamba-DETR.

Runs two parallel window-attention branches at small (s_w) and large (s_fw)
window sizes, bridged by a Dual Local-Global Perception Attention (DLGPA)
mechanism with multi-kernel depthwise convolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers: window partition / reverse
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition (B, H, W, C) into non-overlapping windows of size window_size.
    Returns:
        windows : (num_windows*B, window_size, window_size, C)
        (Hp, Wp) : padded H and W
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_reverse(windows: torch.Tensor, window_size: int,
                   Hp: int, Wp: int, H: int, W: int):
    """
    Reverse window_partition.
    windows : (num_windows*B, window_size, window_size, C)
    Returns : (B, H, W, C)
    """
    B_n = windows.shape[0]
    B = B_n // (Hp // window_size * Wp // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, -1)
    return x[:, :H, :W, :].contiguous()


# ---------------------------------------------------------------------------
# Window-based multi-head self-attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm(x)

        windows, (Hp, Wp) = window_partition(x, self.window_size)
        Bw, ws, _, _ = windows.shape
        x_win = windows.view(Bw, ws * ws, C)

        qkv = self.qkv(x_win).reshape(Bw, ws * ws, 3, self.num_heads,
                                       C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_win = (attn @ v).transpose(1, 2).reshape(Bw, ws * ws, C)
        x_win = self.proj_drop(self.proj(x_win))
        windows = x_win.view(Bw, ws, ws, C)

        x = window_reverse(windows, self.window_size, Hp, Wp, H, W)
        return x + shortcut


# ---------------------------------------------------------------------------
# DLGPA: Dual Local-Global Perception Attention
# ---------------------------------------------------------------------------

class DLGPA(nn.Module):
    """
    Dual Local-Global Perception Attention via multi-kernel depthwise conv.
    Kernel sizes k1=3 (local) and k2=5 (semi-global) are applied in parallel
    and merged to bridge small-window and large-window branches.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        x_c = x.permute(0, 3, 1, 2)              # (B, C, H, W)
        y3 = self.dw3(x_c)
        y5 = self.dw5(x_c)
        y = self.pw(torch.cat([y3, y5], dim=1))   # (B, C, H, W)
        y = y.permute(0, 2, 3, 1)                 # (B, H, W, C)
        return self.act(self.norm(y)) + x


# ---------------------------------------------------------------------------
# Full DSWA Module
# ---------------------------------------------------------------------------

class DSWAModule(nn.Module):
    """
    Dual-Scale Window Attention Module.

    Two parallel WindowAttention branches:
      - small window (s_w=7): fine-grained local patterns
      - large window (s_fw=14): broader contextual patterns

    A DLGPA layer with multi-kernel convolution bridges the two scales.

    Args:
        dim        : channel dimension of the *concatenated* multi-scale feature
        small_win  : small window size (default 7)
        large_win  : large window size (default 14)
        num_heads  : attention heads
    """
    def __init__(self, dim: int, small_win: int = 7, large_win: int = 14,
                 num_heads: int = 8, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.small_attn = WindowAttention(dim, small_win, num_heads,
                                          attn_drop=attn_drop, proj_drop=proj_drop)
        self.large_attn = WindowAttention(dim, large_win, num_heads,
                                          attn_drop=attn_drop, proj_drop=proj_drop)
        self.dlgpa = DLGPA(dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        # Small-window branch
        xs = self.small_attn(x)                             # (B, H, W, C)
        # Large-window branch
        xl = self.large_attn(x)                             # (B, H, W, C)
        # DLGPA bridge
        bridge = self.dlgpa(x)                              # (B, H, W, C)
        # Fuse: small + large; bridge modulates the merged feature
        merged = self.fusion(torch.cat([xs, xl], dim=-1))   # (B, H, W, C)
        out = self.out_proj(merged + bridge)                # (B, H, W, C)
        return out + x                                      # residual
