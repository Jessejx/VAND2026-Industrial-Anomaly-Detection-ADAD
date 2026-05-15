import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHead(nn.Module):
    """Residual anomaly head migrated from DiffFusionModule in main_INP_V5.py."""

    def __init__(self, in_channels: int = 768, hidden_channels: int = 128, out_channels: int = 1):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=2, stride=2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 2, kernel_size=2, stride=2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(hidden_channels // 2, hidden_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels // 4, hidden_channels // 4, kernel_size=2, stride=2),
        )
        self.classifier = nn.Conv2d(hidden_channels // 4, out_channels, kernel_size=1)

    def forward(self, residual_feature: torch.Tensor, out_size=None) -> torch.Tensor:
        x = self.up1(residual_feature)
        x = self.up2(x)
        x = self.up3(x)
        x = self.classifier(x)
        if out_size is not None and x.shape[-2:] != tuple(out_size):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x
