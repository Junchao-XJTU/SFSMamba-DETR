"""
Evaluation script for SFSMamba-DETR.
Computes COCO-style mAP (mAP@50, mAP@50:95) and reports FPS.

Usage:
    python evaluate.py --config configs/sfsmamba_detr_mar20.yaml \\
                       --checkpoint outputs/epoch_149.pth
"""
import argparse
import time
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from models.sfsmamba_detr import SFSMambaDETR
from datasets.base import RemoteSensingDetDataset, collate_fn
from utils.box_ops import box_cxcywh_to_xyxy

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except ImportError:
    HAS_COCO = False


def parse_args():
    parser = argparse.ArgumentParser('SFSMamba-DETR evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--iou_thresh', type=float, default=0.5)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device, conf_thresh, iou_thresh, img_size):
    model.eval()
    results = []
    total_time = 0.0
    total_imgs = 0

    for images, targets in loader:
        images = images.to(device)
        t0 = time.perf_counter()
        detections = model.predict(images, conf_thresh, iou_thresh)
        total_time += time.perf_counter() - t0
        total_imgs += images.shape[0]

        for det, tgt in zip(detections, targets):
            img_id = tgt['image_id']
            boxes = det['boxes']   # xyxy pixel coords
            scores = det['scores']
            labels = det['labels']
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label) + 1,   # COCO uses 1-based ids
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(score),
                })

    fps = total_imgs / total_time
    print(f'FPS: {fps:.1f}  ({total_imgs} images in {total_time:.2f}s)')

    if HAS_COCO and results:
        import json, tempfile
        ann_file = loader.dataset.ann_file if hasattr(loader.dataset, 'ann_file') else None
        if ann_file:
            coco_gt = COCO(ann_file)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as fp:
                json.dump(results, fp)
                tmp = fp.name
            coco_dt = coco_gt.loadRes(tmp)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    return fps


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SFSMambaDETR(
        num_classes=cfg['num_classes'],
        d_model=cfg.get('d_model', 256),
        num_queries=cfg.get('num_queries', 300),
        num_decoder_layers=cfg.get('num_decoder_layers', 3),
        small_win=cfg.get('small_win', 7),
        large_win=cfg.get('large_win', 14),
        pretrained_backbone=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    val_dataset = RemoteSensingDetDataset(
        img_dir=cfg['val_img_dir'],
        ann_file=cfg['val_ann_file'],
        img_size=cfg.get('img_size', 640),
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.get('val_batch_size', 8),
        shuffle=False, num_workers=cfg.get('num_workers', 4),
        collate_fn=collate_fn, pin_memory=True,
    )

    evaluate(model, val_loader, device,
             args.conf_thresh, args.iou_thresh, cfg.get('img_size', 640))


if __name__ == '__main__':
    main()
