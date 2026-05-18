"""
2D Selective Scan (SS2D) module for SFSMamba-DETR.
Scans feature maps in four directions (H+, H-, V+, V-) using
the Mamba selective state space mechanism.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


def selective_scan_ref(u, delta, A, B, C, D=None):
    """
    Pure-PyTorch reference implementation of selective scan.
    u     : (B, D, L)
    delta : (B, D, L)
    A     : (D, N)
    B     : (B, N, L)
    C     : (B, N, L)
    D     : (D,)  optional skip connection
    Returns y : (B, D, L)
    """
    B_b, D_d, L = u.shape
    N = A.shape[1]
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()

    delta = F.softplus(delta)                                      # (B, D, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))    # (B, D, L, N)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)    # (B, D, L, N)

    h = torch.zeros(B_b, D_d, N, device=u.device)
    ys = []
    for i in range(L):
        h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
        y = torch.einsum('bdn,bn->bd', h, C[:, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)   # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y.to(dtype_in)


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    selective_scan_fn = selective_scan_ref
    HAS_MAMBA = False


class SS2D(nn.Module):
    """
    2D Selective Scan with four-directional scanning.
    Input/output shape: (B, H, W, C).
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 dt_rank: str = "auto", d_conv: int = 3,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=bias
        )
        self.act = nn.SiLU()

        # Per-direction projection weights (4 dirs share one Linear for memory efficiency)
        self.x_proj = nn.Linear(
            self.d_inner, (self.dt_rank + self.d_state * 2) * 4, bias=False
        )
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(4)
        ])
        # Init dt_proj bias (from Mamba paper)
        for dt_proj in self.dt_projs:
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

        # A matrices (4 directions × D × N)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32),
                   'n -> d n', d=self.d_inner)
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(A.clone())) for _ in range(4)
        ])
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(4)
        ])

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _scan_direction(self, x_dir, idx):
        """Run SSM for one direction. x_dir: (B, D, L)"""
        B, D, L = x_dir.shape
        # Project to dt, B, C
        xz = self.x_proj(x_dir.permute(0, 2, 1))          # (B, L, dt_rank+2N)
        xz_split = xz.chunk(4, dim=-1)[0]                  # use first split (already direction-indexed via loop)
        xz_dir = self.x_proj(x_dir.permute(0, 2, 1))
        dt_rank, N = self.dt_rank, self.d_state
        dt, B_mat, C_mat = xz_dir.split([dt_rank, N, N], dim=-1)
        dt = self.dt_projs[idx](dt).permute(0, 2, 1)       # (B, D, L)
        B_mat = B_mat.permute(0, 2, 1)                     # (B, N, L)
        C_mat = C_mat.permute(0, 2, 1)                     # (B, N, L)
        A = -torch.exp(self.A_logs[idx].float())            # (D, N)
        D = self.Ds[idx].float()

        y = selective_scan_fn(x_dir, dt, A, B_mat, C_mat, D)
        return y   # (B, D, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) → (B, H, W, C)"""
        B, H, W, C = x.shape
        xz = self.in_proj(x)                               # (B, H, W, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                      # each (B, H, W, d_inner)

        x_in = self.act(self.conv2d(x_in.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)

        # Prepare four directional sequences
        seqs = [
            x_in.reshape(B, H * W, -1).permute(0, 2, 1),          # H+ row-major
            x_in.flip(1).reshape(B, H * W, -1).permute(0, 2, 1),  # H- (flip H)
            x_in.permute(0, 2, 1, 3).reshape(B, H * W, -1).permute(0, 2, 1),          # V+
            x_in.permute(0, 2, 1, 3).flip(1).reshape(B, H * W, -1).permute(0, 2, 1),  # V-
        ]

        ys = []
        for i, seq in enumerate(seqs):
            y = self._scan_direction(seq, i)                # (B, D, L)
            ys.append(y.permute(0, 2, 1))                  # (B, L, D)

        # Reverse flips and sum
        y0 = ys[0].reshape(B, H, W, -1)
        y1 = ys[1].reshape(B, H, W, -1).flip(1)
        y2 = ys[2].reshape(B, H, W, -1).permute(0, 2, 1, 3)
        y3 = ys[3].flip(1).reshape(B, H, W, -1).permute(0, 2, 1, 3)

        y = y0 + y1 + y2 + y3
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y)
