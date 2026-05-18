"""
Training script for SFSMamba-DETR.

Usage:
    python train.py --config configs/sfsmamba_detr_mar20.yaml
"""
import argparse
import os
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.sfsmamba_detr import SFSMambaDETR
from utils.losses import SFSMambaLoss
from datasets.base import RemoteSensingDetDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser('SFSMamba-DETR training')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    model = SFSMambaDETR(
        num_classes=cfg['num_classes'],
        d_model=cfg.get('d_model', 256),
        num_queries=cfg.get('num_queries', 300),
        num_decoder_layers=cfg.get('num_decoder_layers', 3),
        small_win=cfg.get('small_win', 7),
        large_win=cfg.get('large_win', 14),
        d_state=cfg.get('d_state', 16),
        num_heads=cfg.get('num_heads', 8),
        dim_feedforward=cfg.get('dim_feedforward', 1024),
        dropout=cfg.get('dropout', 0.0),
        pretrained_backbone=cfg.get('pretrained_backbone', True),
    )
    return model.to(device)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset = RemoteSensingDetDataset(
        img_dir=cfg['train_img_dir'],
        ann_file=cfg['train_ann_file'],
        img_size=cfg.get('img_size', 640),
        augment=True,
    )
    val_dataset = RemoteSensingDetDataset(
        img_dir=cfg['val_img_dir'],
        ann_file=cfg['val_ann_file'],
        img_size=cfg.get('img_size', 640),
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.get('batch_size', 16),
        shuffle=True, num_workers=cfg.get('num_workers', 4),
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.get('val_batch_size', 8),
        shuffle=False, num_workers=cfg.get('num_workers', 4),
        collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = build_model(cfg, device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = SFSMambaLoss(
        num_classes=cfg['num_classes'],
        lambda_cls=cfg.get('lambda_cls', 2.0),
        lambda_l1=cfg.get('lambda_l1', 5.0),
        lambda_giou=cfg.get('lambda_giou', 2.0),
        lambda_iou=cfg.get('lambda_iou', 1.0),
    ).to(device)

    # Different LR for backbone vs rest
    backbone_params = list(
        (model.module if hasattr(model, 'module') else model).backbone.parameters()
    )
    other_params = [
        p for p in model.parameters()
        if not any(p is bp for bp in backbone_params)
    ]
    optimizer = AdamW(
        [{'params': backbone_params, 'lr': cfg.get('lr', 1e-4) * 0.1},
         {'params': other_params,   'lr': cfg.get('lr', 1e-4)}],
        weight_decay=cfg.get('weight_decay', 1e-4),
    )
    epochs = cfg.get('epochs', 150)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            cls_logits, bbox, iou = model(images)
            outputs = {'cls_logits': cls_logits, 'bbox': bbox, 'iou': iou}
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss_total']

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'(cls={loss_dict["loss_cls"]:.3f} '
                      f'l1={loss_dict["loss_l1"]:.3f} '
                      f'giou={loss_dict["loss_giou"]:.3f})')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} avg loss: {avg_loss:.4f}')

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)

    print('Training complete.')


if __name__ == '__main__':
    main()
