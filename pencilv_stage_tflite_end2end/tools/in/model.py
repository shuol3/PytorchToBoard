from typing import List

import torch.nn as nn


class MelCNN(nn.Module):
    """
    轻量级 log-mel CNN。

    相比原始版本，这里做了三点小改动：
    1. Conv 后加入 BatchNorm2d，让训练更稳定；
    2. 支持 2 层或 3 层卷积，通过 config 里的 channels 控制；
    3. 仍然使用 AdaptiveAvgPool2d，避免手动计算展平尺寸。
    """

    def __init__(self, num_classes: int, channels: List[int], hidden_dim: int, dropout: float):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("model.channels 至少需要两个通道数，例如 [16, 32] 或 [16, 32, 64]")

        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

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
