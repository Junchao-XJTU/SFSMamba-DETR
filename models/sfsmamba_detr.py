"""
SFSMamba-DETR: Full model definition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.efficientnet import EfficientNetBackbone
from .encoder.aifi import AIFIModule
from .encoder.cfam import CFAMEncoder
from .decoder.decoder import TransformerDecoder


class SFSMambaDETR(nn.Module):
    """
    SFSMamba-DETR: Selective Feature Scanning with State Space Models
    and Dual-Scale Window Attention for Remote Sensing Object Detection.

    Architecture:
      1. EfficientNet-B4 backbone  → {P2, P3, P4, P5}
      2. AIFI on P5                → F5_enhanced
      3. CFAM (SFS + FusionBlocks + DSWA) → aggregated feature
      4. TransformerDecoder        → (cls, bbox, iou)
    """
    def __init__(
        self,
        num_classes: int = 80,
        d_model: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 3,
        small_win: int = 7,
        large_win: int = 14,
        d_state: int = 16,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = EfficientNetBackbone(
            model_name='efficientnet_b4', pretrained=pretrained_backbone
        )
        in_dims = self.backbone.out_channels   # e.g. [32, 56, 160, 448]

        # ── AIFI on P5 ────────────────────────────────────────────────────
        self.p5_proj = nn.Sequential(
            nn.Conv2d(in_dims[3], d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.aifi = AIFIModule(
            d_model=d_model, num_heads=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout
        )

        # ── CFAM (encoder) ────────────────────────────────────────────────
        self.cfam = CFAMEncoder(
            in_dims=in_dims[:3],    # [P2, P3, P4] channels
            dim=d_model,
            small_win=small_win,
            large_win=large_win,
            d_state=d_state,
            num_heads=num_heads,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            num_classes=num_classes,
            num_queries=num_queries,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)
        Returns:
            cls_logits : (B, num_queries, num_classes)
            bbox       : (B, num_queries, 4) – [cx, cy, w, h] in [0,1]
            iou        : (B, num_queries, 1)
        """
        B, _, H, W = x.shape

        # 1. Backbone
        feats = self.backbone(x)         # [P2, P3, P4, P5]

        # 2. AIFI on P5
        f5 = self.p5_proj(feats[3])      # (B, d_model, H5, W5)
        f5 = self.aifi(f5)               # (B, d_model, H5, W5)

        # 3. CFAM: passes [P2, P3, P4] and enhanced F5
        enc_out = self.cfam(feats[:3], f5)    # (B, d_model, H3, W3)

        # Flatten for decoder
        B, C, Hf, Wf = enc_out.shape
        memory = enc_out.flatten(2).transpose(1, 2)   # (B, L, C)

        # 4. Decode
        cls_logits, bbox, iou = self.decoder(memory)
        return cls_logits, bbox, iou

    @torch.no_grad()
    def predict(self, x: torch.Tensor, conf_thresh: float = 0.3,
                iou_thresh: float = 0.5):
        """
        Run inference and return filtered detections.
        Returns list of dicts with keys: 'boxes', 'scores', 'labels'
        """
        from torchvision.ops import batched_nms
        cls_logits, bbox, iou = self(x)
        scores_all = cls_logits.sigmoid() * iou   # (B, Q, nc)
        results = []
        for b in range(x.shape[0]):
            scores, labels = scores_all[b].max(-1)    # (Q,)
            keep = scores > conf_thresh
            boxes = bbox[b][keep]       # (k, 4) cx/cy/w/h
            s = scores[keep]
            l = labels[keep]

            # Convert cx/cy/w/h to x1/y1/x2/y2
            x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
            x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
            xyxy = torch.cat([x1y1, x2y2], dim=-1)
            # Scale to pixel coords
            xyxy *= torch.tensor([x.shape[3], x.shape[2],
                                   x.shape[3], x.shape[2]],
                                  device=x.device, dtype=boxes.dtype)

            keep2 = batched_nms(xyxy, s, l, iou_thresh)
            results.append({
                'boxes': xyxy[keep2],
                'scores': s[keep2],
                'labels': l[keep2],
            })
        return results
