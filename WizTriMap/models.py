import time
import numpy as np
from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.optim as optim


def combined_ssim_mse_loss(output, target, alpha=0.5):
    mse = nn.functional.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return alpha * mse + (1 - alpha) * ssim_loss

class InversionMLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class InversionMLP_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 3072), nn.Sigmoid()  # 3 × 32 × 32
        )

    def forward(self, x):
        return self.model(x)

class ConvDecoderGray(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128 * 7 * 7),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),           # smoothing
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 128, 7, 7)
        return self.deconv(x)

class ConvDecoder(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 128),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x).view(-1, 128, 8, 8)
        return self.deconv(x)

class ConvDecoderGrayV2(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128 * 7 * 7),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 128, 7, 7)
        return self.deconv(x)

class ConvDecoderV2(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 8 * 8 * 256),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 256, 8, 8)
        return self.deconv(x)

class ConvDecoderV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 8*8*512),
            nn.BatchNorm1d(8*8*512),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            # Block 1: 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            
            # Block 2: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Block 3: Feature refinement
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Final output
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        return self.decoder(x)