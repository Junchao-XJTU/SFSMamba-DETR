# SFSMamba-DETR

**SFSMamba-DETR: Selective Feature Scanning with State Space Models and Dual-Scale Window Attention for Remote Sensing Object Detection**

> Junchao Zhao, Yuanli Cai, Husheng Wu, Rui Ma  
> School of Automation Science and Engineering, Xi'an Jiaotong University

---

## Overview

SFSMamba-DETR is a DETR-style object detector tailored for aerial and satellite remote sensing imagery. It integrates:

| Module | Role |
|--------|------|
| **SFS** (Selective Feature Scanning) | Tri-input cross-scale feature fusion via 2D selective scan (SS2D) |
| **DSWA** (Dual-Scale Window Attention) | Parallel small/large window attention with DLGPA multi-kernel bridge |
| **CFAM** (Cross-scale Feature Aggregation Module) | Hierarchical top-down encoder built from SFS + Fusion blocks |
| **AIFI** | Attention-based Intra-scale Feature Interaction on P5 |
| **EfficientNet-B4** | Compound-scaled CNN backbone |
| **IoU-aware query selection** | Selects high-quality decoder queries from encoder proposals |

### Architecture

```
Input image (640×640)
    │
    ▼
EfficientNet-B4 backbone  →  [P2, P3, P4, P5]
                                          │
                                     AIFI(P5)  →  F5
                                          │
                     CFAM: SFS(P4,F5,P3) → F4
                           SFS(P3,F4,P2) → F3
                           DSWA(F3+F4+F5) → encoder output
                                          │
                          TransformerDecoder (3 layers, 300 queries)
                                          │
                              (cls_logits, bbox, iou)
```

---

## Installation

```bash
git clone https://github.com/<your-repo>/SFSMamba-DETR.git
cd SFSMamba-DETR
pip install -r requirements.txt
```

Optional: install the CUDA-accelerated Mamba kernel for faster training (Linux + CUDA 11.6+):
```bash
pip install mamba-ssm
```

---

## Datasets

Prepare datasets in COCO JSON format and update `configs/*.yaml` with the correct paths.

| Dataset | Classes | URL |
|---------|---------|-----|
| **MAR20** | 20 aircraft types | https://gcheng-nwpu.github.io/ |
| **UCAS-AOD** | car, airplane | https://github.com/Lbx2020/UCAS-AOD-dataset |
| **Jilin-1** | aircraft | https://www.jl1mall.com/ |

Expected directory layout:
```
data/
  MAR20/
    images/{train,val}/
    annotations/instances_{train,val}.json
  UCAS-AOD/
    images/{train,val}/
    annotations/instances_{train,val}.json
  Jilin1/
    images/{train,val}/
    annotations/instances_{train,val}.json
```

---

## Training

```bash
python train.py --config configs/sfsmamba_detr_mar20.yaml --output_dir outputs/mar20
```

Resume from checkpoint:
```bash
python train.py --config configs/sfsmamba_detr_mar20.yaml \
                --resume outputs/mar20/epoch_050.pth
```

### Hyper-parameters (paper defaults)

| Parameter | Value |
|-----------|-------|
| Input size | 640 × 640 |
| Epochs | 150 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-4 (backbone ×0.1) |
| Weight decay | 1e-4 |
| LR schedule | Cosine annealing |
| λ_cls / λ_L1 / λ_GIoU / λ_IoU | 2.0 / 5.0 / 2.0 / 1.0 |
| DSWA small/large window | 7 / 14 |
| Decoder layers / queries | 3 / 300 |

---

## Evaluation

```bash
python evaluate.py --config configs/sfsmamba_detr_mar20.yaml \
                   --checkpoint outputs/mar20/epoch_149.pth
```

---

## Results

| Dataset | mAP@50 | mAP@50:95 | FPS |
|---------|--------|-----------|-----|
| MAR20 | — | — | — |
| UCAS-AOD | — | — | — |
| Jilin-1 | — | — | — |

*(Fill in after training.)*

---

## Citation

```bibtex
@article{zhao2026sfsmamba,
  title   = {SFSMamba-DETR: Selective Feature Scanning with State Space Models 
             and Dual-Scale Window Attention for Remote Sensing Object Detection},
  author  = {Zhao, Junchao and Cai, Yuanli and Wu, Husheng and Ma, Rui},
  journal = {Remote Sensing},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License.
