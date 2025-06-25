from pytorch_msssim import ssim
import torch
import torch.nn as nn


def combined_ssim_mse_loss(output, target, alpha=0.5):
    mse = nn.functional.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return alpha * mse + (1 - alpha) * ssim_loss


class ConvDecoderGray(nn.Module):
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


class ConvDecoder(nn.Module):
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


def get_inversion_from_model(model, x_embedding, sample_size, device):
    embedding_tensor = torch.tensor(x_embedding, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        x_recon = model(embedding_tensor).cpu().numpy()
    x_recon = x_recon.reshape(sample_size, -1)
    return x_recon
