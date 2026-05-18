"""
Loss functions for SFSMamba-DETR.
Includes Hungarian matching + focal classification + L1 bbox + GIoU + IoU losses.
λ1=2.0 (cls), λ2=5.0 (L1), λ3=2.0 (GIoU), λ4=1.0 (IoU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list):
        """
        outputs : {'cls_logits': (B,Q,nc), 'bbox': (B,Q,4)}
        targets : list of dicts with 'labels' (n,) and 'boxes' (n,4) in cxcywh [0,1]
        Returns list of (row_ind, col_ind) per image.
        """
        B, Q, nc = outputs['cls_logits'].shape
        pred_cls = outputs['cls_logits'].flatten(0, 1).sigmoid()   # (B*Q, nc)
        pred_bbox = outputs['bbox'].flatten(0, 1)                  # (B*Q, 4)

        sizes = [len(t['labels']) for t in targets]
        tgt_cls = torch.cat([t['labels'] for t in targets])        # (sum_n,)
        tgt_bbox = torch.cat([t['boxes'] for t in targets])        # (sum_n, 4)

        # Classification cost
        cost_class = -pred_cls[:, tgt_cls]                         # (B*Q, sum_n)

        # L1 cost
        cost_bbox = torch.cdist(pred_bbox, tgt_bbox, p=1)          # (B*Q, sum_n)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )                                                           # (B*Q, sum_n)

        C = (self.cost_class * cost_class
             + self.cost_bbox * cost_bbox
             + self.cost_giou * cost_giou)
        C = C.view(B, Q, -1).cpu()

        indices = []
        offset = 0
        for i, s in enumerate(sizes):
            c = C[i, :, offset:offset + s]
            row, col = linear_sum_assignment(c.numpy())
            indices.append((torch.as_tensor(row, dtype=torch.int64),
                            torch.as_tensor(col, dtype=torch.int64)))
            offset += s
        return indices


class SFSMambaLoss(nn.Module):
    def __init__(self, num_classes: int,
                 lambda_cls: float = 2.0, lambda_l1: float = 5.0,
                 lambda_giou: float = 2.0, lambda_iou: float = 1.0,
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_iou = lambda_iou
        self.alpha = alpha
        self.gamma = gamma
        self.matcher = HungarianMatcher(lambda_cls, lambda_l1, lambda_giou)

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary focal loss."""
        p = pred.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        fl = alpha_t * ((1 - p_t) ** self.gamma) * ce
        return fl.mean()

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        outputs : {'cls_logits': (B,Q,nc), 'bbox': (B,Q,4), 'iou': (B,Q,1)}
        targets : list of {'labels':(n,), 'boxes':(n,4)} per image
        """
        indices = self.matcher(outputs, targets)

        cls_logits = outputs['cls_logits']   # (B, Q, nc)
        bbox_pred = outputs['bbox']          # (B, Q, 4)
        iou_pred = outputs['iou']            # (B, Q, 1)

        B = cls_logits.shape[0]
        device = cls_logits.device

        loss_cls = torch.tensor(0., device=device)
        loss_l1 = torch.tensor(0., device=device)
        loss_giou = torch.tensor(0., device=device)
        loss_iou = torch.tensor(0., device=device)
        num_boxes = max(1, sum(len(t['labels']) for t in targets))

        for i, (row, col) in enumerate(indices):
            if len(row) == 0:
                continue
            # Classification
            tgt_cls_oh = torch.zeros_like(cls_logits[i])
            tgt_labels = targets[i]['labels'][col].to(device)
            tgt_cls_oh[row, tgt_labels] = 1.0
            loss_cls += self.focal_loss(cls_logits[i], tgt_cls_oh)

            # BBox
            pred_b = bbox_pred[i][row]             # (k, 4)
            tgt_b = targets[i]['boxes'][col].to(device)   # (k, 4)
            loss_l1 += F.l1_loss(pred_b, tgt_b, reduction='sum') / num_boxes

            # GIoU
            pred_xyxy = box_cxcywh_to_xyxy(pred_b)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_b)
            giou = torch.diag(generalized_box_iou(pred_xyxy, tgt_xyxy))
            loss_giou += (1 - giou).sum() / num_boxes

            # IoU supervision
            iou_gt = torch.diag(box_iou(pred_xyxy, tgt_xyxy)).unsqueeze(-1)
            loss_iou += F.binary_cross_entropy(
                iou_pred[i][row], iou_gt, reduction='sum'
            ) / num_boxes

        total = (self.lambda_cls * loss_cls
                 + self.lambda_l1 * loss_l1
                 + self.lambda_giou * loss_giou
                 + self.lambda_iou * loss_iou)

        return {
            'loss_total': total,
            'loss_cls': loss_cls.detach(),
            'loss_l1': loss_l1.detach(),
            'loss_giou': loss_giou.detach(),
            'loss_iou': loss_iou.detach(),
        }
