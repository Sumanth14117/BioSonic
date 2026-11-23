import torch
from torch import nn

from . import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout))

    def forward(self, x):
        return self.layers(x)


class BioSonicCNN(nn.Module):
    """Custom four-stage CNN for [batch, 1, 128 Mel bands, 622 frames]."""

    def __init__(self, num_classes=len(config.INDEX_TO_LABEL), channels=(16, 32, 64, 128), dropout=0.12,
                 classifier_dropout=0.40):
        super().__init__()
        blocks = []
        incoming = 1
        for level, outgoing in enumerate(channels):
            blocks.append(ConvBlock(incoming, outgoing, dropout + 0.03 * level))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], 128), nn.ReLU(inplace=True),
                                        nn.Dropout(classifier_dropout), nn.Linear(128, num_classes))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != config.N_MELS:
            raise ValueError("Expected input [batch, 1, 128, time_frames]")
        return self.classifier(self.pool(self.features(x)))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def architecture_config(model=None):
    model = model or BioSonicCNN()
    return {"architecture": "Custom BioSonicCNN; four ConvBlock stages; no pretrained weights",
            "input_shape": [1, config.N_MELS, 622], "channels": [16, 32, 64, 128],
            "block": "two Conv2d(3x3)-BatchNorm-ReLU layers, MaxPool2d(2), Dropout2d",
            "pooling": "AdaptiveAvgPool2d((1,1))", "classifier": "128-unit Linear-ReLU-Dropout-8 logits",
            "num_classes": len(config.INDEX_TO_LABEL), "class_order": list(config.INDEX_TO_LABEL),
            "trainable_parameters": parameter_count(model), "output": "raw logits; CrossEntropyLoss"}
