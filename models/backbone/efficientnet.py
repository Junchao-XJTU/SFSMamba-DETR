"""
EfficientNet-B4 backbone wrapper that returns a 4-level feature pyramid
{P2, P3, P4, P5} at strides {4, 8, 16, 32}.
"""
import torch
import torch.nn as nn

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class EfficientNetBackbone(nn.Module):
    """
    Wraps timm's EfficientNet-B4 to expose multi-scale features.
    Output channels (before FPN projection): [32, 56, 160, 448] for B4.
    """
    # EfficientNet-B4 feature_info indices for strides [4, 8, 16, 32]
    OUT_INDICES = (1, 2, 3, 4)
    OUT_CHANNELS = {'efficientnet_b4': [32, 56, 160, 448]}

    def __init__(self, model_name: str = 'efficientnet_b4',
                 pretrained: bool = True, out_indices: tuple = None):
        super().__init__()
        assert HAS_TIMM, "timm is required: pip install timm"
        indices = out_indices or self.OUT_INDICES
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=indices,
        )
        feat_info = self.backbone.feature_info.channels()
        self.out_channels = list(feat_info)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 3, H, W)
        Returns list of tensors [P2, P3, P4, P5] with strides [4,8,16,32]
        """
        return self.backbone(x)
