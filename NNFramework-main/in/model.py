"""定义用于 log-mel 特征分类的轻量 CNN 模型。"""

from __future__ import annotations

from typing import List

import torch.nn as nn


class MelCNN(nn.Module):
    """面向固定输入特征尺寸的轻量音频分类网络。"""

    def __init__(
        self,
        num_classes: int,
        channels: List[int],
        hidden_dim: int,
        dropout: float,
        feature_bins: int = 48,
        feature_frames: int = 94,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("model.channels 至少需要两个通道数，例如 [16, 32] 或 [16, 32, 64]")

        pooled_bins = feature_bins
        pooled_frames = feature_frames
        layers = []
        in_channels = 1

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            pooled_bins //= 2
            pooled_frames //= 2
            in_channels = out_channels

        if pooled_bins <= 0 or pooled_frames <= 0:
            raise ValueError("当前输入尺寸在多次池化后已经无效，请减小池化层数或增大输入特征尺寸")

        # 使用固定 AveragePool2d，尽量让导出图落到 AVERAGE_POOL_2D 而不是 MEAN。
        layers.append(nn.AvgPool2d(kernel_size=(pooled_bins, pooled_frames)))
        self.features = nn.Sequential(*layers)

        # 分类头保持不变，方便复用已有 checkpoint 权重。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
