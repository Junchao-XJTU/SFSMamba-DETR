"""
Base dataset class and shared transforms for remote sensing detection.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random


class RemoteSensingDetDataset(Dataset):
    """
    Base dataset. Expects a COCO-style JSON annotation file.
    Each image entry must have 'file_name', and annotations with
    'bbox' (x,y,w,h pixels) and 'category_id'.
    """
    def __init__(self, img_dir: str, ann_file: str,
                 img_size: int = 640, augment: bool = False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.imgs = {img['id']: img for img in coco['images']}
        self.img_ids = [img['id'] for img in coco['images']]
        self.cat_map = {c['id']: i for i, c in enumerate(coco['categories'])}
        self.num_classes = len(coco['categories'])

        # Group annotations by image id
        self.anns = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            self.anns[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def _load_image(self, img_id: int) -> Image.Image:
        info = self.imgs[img_id]
        path = os.path.join(self.img_dir, info['file_name'])
        img = Image.open(path).convert('RGB')
        return img

    def _load_target(self, img_id: int, orig_w: int, orig_h: int):
        """Returns boxes in cxcywh normalised [0,1] and labels."""
        boxes, labels = [], []
        for ann in self.anns[img_id]:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            boxes.append([cx, cy, w / orig_w, h / orig_h])
            labels.append(self.cat_map[ann['category_id']])
        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), \
                   torch.tensor(labels, dtype=torch.int64)
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.int64)

    def _transform(self, img: Image.Image):
        """Resize to img_size × img_size with optional augmentations."""
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])
        return tensor

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img = self._load_image(img_id)
        orig_w, orig_h = img.size
        boxes, labels = self._load_target(img_id, orig_w, orig_h)
        img_tensor = self._transform(img)
        return img_tensor, {'boxes': boxes, 'labels': labels, 'image_id': img_id}


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
