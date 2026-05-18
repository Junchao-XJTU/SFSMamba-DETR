"""
RT-DETR-style decoder for SFSMamba-DETR.
Includes IoU-aware query selection and separable dynamic decoder heads.
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


def get_activation(name: str) -> nn.Module:
    return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU()}[name]


# ---------------------------------------------------------------------------
# IoU-aware query selection
# ---------------------------------------------------------------------------

class IoUAwareQuerySelection(nn.Module):
    """
    Select the top-k encoder proposals as decoder queries based on
    predicted IoU score × classification confidence.
    """
    def __init__(self, num_queries: int = 300, in_channels: int = 256,
                 num_classes: int = 80):
        super().__init__()
        self.num_queries = num_queries
        self.cls_head = nn.Linear(in_channels, num_classes)
        self.iou_head = nn.Linear(in_channels, 1)
        self.bbox_head = nn.Linear(in_channels, 4)

    def forward(self, features: torch.Tensor):
        """
        features : (B, L, C) – flattened encoder output
        Returns:
            queries      : (B, num_queries, C)
            ref_points   : (B, num_queries, 4) – reference boxes [cx,cy,w,h] in [0,1]
        """
        cls_logits = self.cls_head(features)            # (B, L, num_classes)
        iou_pred = self.iou_head(features).sigmoid()    # (B, L, 1)
        bbox_pred = self.bbox_head(features).sigmoid()  # (B, L, 4)

        # Score = max cls prob × IoU
        score = cls_logits.sigmoid().max(-1).values * iou_pred.squeeze(-1)
        topk = score.topk(self.num_queries, dim=1).indices   # (B, num_queries)

        queries = features.gather(1, topk.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        ref_points = bbox_pred.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4))
        return queries, ref_points


# ---------------------------------------------------------------------------
# Separable dynamic decoder head
# ---------------------------------------------------------------------------

class SeparableDynamicHead(nn.Module):
    """
    Classification + bounding-box + IoU prediction heads
    with layer-norm and residual connections.
    """
    def __init__(self, d_model: int, num_classes: int, num_layers: int = 3):
        super().__init__()
        self.cls_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
            for _ in range(num_layers)
        ])
        self.bbox_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
            for _ in range(num_layers)
        ])
        self.cls_head = nn.Linear(d_model, num_classes)
        self.bbox_head = nn.Linear(d_model, 4)
        self.iou_head = nn.Linear(d_model, 1)

    def forward(self, q: torch.Tensor, ref: torch.Tensor):
        """
        q   : (B, Q, C)
        ref : (B, Q, 4) – reference boxes
        Returns: cls_logits (B,Q,num_classes), bbox (B,Q,4), iou (B,Q,1)
        """
        cls_q = q
        for layer in self.cls_layers:
            cls_q = layer(cls_q) + cls_q

        box_q = q
        for layer in self.bbox_layers:
            box_q = layer(box_q) + box_q

        cls_logits = self.cls_head(cls_q)                   # (B, Q, nc)
        bbox_delta = self.bbox_head(box_q).sigmoid()        # (B, Q, 4)
        iou = self.iou_head(box_q).sigmoid()                # (B, Q, 1)

        # Refine reference boxes with predicted delta
        ref_inv = inverse_sigmoid(ref)
        bbox = (ref_inv + bbox_delta).sigmoid()             # (B, Q, 4)
        return cls_logits, bbox, iou


# ---------------------------------------------------------------------------
# Decoder layer (cross-attention with encoder memory)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """q: (B,Q,C), memory: (B,L,C)"""
        # Self-attention
        q2 = self.norm1(q)
        q2, _ = self.self_attn(q2, q2, q2)
        q = q + self.drop1(q2)
        # Cross-attention with encoder memory
        q2 = self.norm2(q)
        q2, _ = self.cross_attn(q2, memory, memory)
        q = q + self.drop2(q2)
        # FFN
        q = q + self.drop3(self.ffn(self.norm3(q)))
        return q


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 3, num_classes: int = 80,
                 num_queries: int = 300, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.query_selection = IoUAwareQuerySelection(num_queries, d_model, num_classes)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.head = SeparableDynamicHead(d_model, num_classes, num_layers)

    def forward(self, memory: torch.Tensor):
        """
        memory : (B, L, C) – flattened CFAM/DSWA output
        Returns:
            cls_logits : (B, Q, num_classes)
            bbox       : (B, Q, 4) in [0,1] cx/cy/w/h
            iou        : (B, Q, 1)
        """
        queries, ref_points = self.query_selection(memory)
        q = queries
        for layer in self.layers:
            q = layer(q, memory)
        cls_logits, bbox, iou = self.head(q, ref_points)
        return cls_logits, bbox, iou
