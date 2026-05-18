"""
AIFI: Attention-based Intra-scale Feature Interaction.
Applied to the highest-level feature P5 before CFAM.
Equivalent to a standard Transformer encoder layer with sinusoidal positional encoding.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) → adds 2D sin/cos positional encoding"""
        B, H, W, C = x.shape
        assert C % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal PE"

        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).expand(H, W)

        dim_t = torch.arange(C // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1)

        pos = torch.cat([pos_y.flatten(-2), pos_x.flatten(-2)], dim=-1)  # (H, W, C)
        return x + pos.unsqueeze(0)


class AIFIModule(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.
    Applies multi-head self-attention on flattened P5 feature.
    """
    def __init__(self, d_model: int, num_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.pe = SinusoidalPositionalEncoding2D(d_model)

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)  (feature map format from CNN backbone)
        Returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)         # (B, H, W, C)
        x = self.pe(x)                     # add 2D PE
        x = x.reshape(B, H * W, C)        # (B, L, C)

        # Self-attention with residual
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.drop1(x2)

        # FFN with residual
        x = x + self.drop2(self.ffn(self.norm2(x)))

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)
        return x
