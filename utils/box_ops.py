"""Bounding box utilities for SFSMamba-DETR."""
import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) → (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    GIoU between two sets of boxes (xyxy format).
    Returns (N, M) matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_area = (rb_enc - lt_enc).clamp(0).prod(-1)

    giou = iou - (enc_area - union) / (enc_area + 1e-6)
    return giou


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes in xyxy format. Returns (N, M) matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)
