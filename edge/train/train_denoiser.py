"""
图像去噪模型训练脚本 (NAFNet-Mobile)
======================================
基于 NAFNet (Nonlinear Activation Free Network) 的轻量去噪模型
在保留细节的同时有效去除噪声，适合低光照/雨天路口场景

使用方法:
    # 使用合成噪声训练 (快速启动)
    python train_denoiser.py --data_dir /path/to/clean_images --noise_type gaussian --epochs 200
    
    # 使用真实噪声数据训练
    python train_denoiser.py --data_dir /path/to/paired_data --noise_type real --epochs 300

数据集格式:
    dataset/
    ├── train/
    │   ├── clean/     # 干净图像
    │   └── noisy/     # 噪声图像 (真实噪声训练时需要)
    └── val/
        ├── clean/
        └── noisy/
"""
import os
import sys
import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required. Install: pip install torch torchvision")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NAFBlock(nn.Module):
    """NAFNet 基础块: 简化的通道注意力 + 残差连接"""
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = c * dw_expand
        
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel, c, 1)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel, dw_channel, 1, 1, 0, groups=1),
        )
        
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * ffn_expand, 1),
            nn.GELU(),
            nn.Conv2d(c * ffn_expand, c, 1)
        )
    
    def forward(self, x):
        inp = x
        
        # LayerNorm 需要 [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.sca(x)  # 简化的通道注意力
        x = self.conv3(x)
        
        y = inp + x
        
        # FFN
        x = y.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.ffn(x)
        
        return y + x


class NAFNetMobile(nn.Module):
    """
    轻量版 NAFNet 去噪网络
    
    相比标准 UNet:
    - 去除了非线性激活函数中的 ReLU/SiLU，减少信息损失
    - 简化的通道注意力，计算量更低
    - 参数量约 2-5M，适合边缘部署
    """
    def __init__(
        self,
        img_channels=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channels, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channels, 3, 1, 1)
        
        # 编码器
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2
        
        # 中间层
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        # 解码器
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
        
        self.padder_size = 2 ** len(enc_blk_nums)
    
    def forward(self, x):
        # 输入填充到 2^n 倍数
        B, C, H, W = x.shape
        pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        
        x = self.intro(x)
        encs = []
        
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x[:, :, :H, :W]  # 去除填充
        
        return x


class DenoiseDataset(Dataset):
    """去噪数据集"""
    def __init__(self, data_dir, split="train", img_size=256, noise_type="gaussian", noise_level=25):
        self.data_dir = Path(data_dir) / split
        self.img_size = img_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        clean_dir = self.data_dir / "clean"
        if clean_dir.exists():
            self.clean_images = sorted(list(clean_dir.glob("*.jpg")) + list(clean_dir.glob("*.png")))
        else:
            self.clean_images = sorted(list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png")))
        
        self.paired = (self.data_dir / "noisy").exists()
        logger.info(f"Denoise {split}: {len(self.clean_images)} images, paired={self.paired}")
    
    def __len__(self):
        return len(self.clean_images)
    
    def add_noise(self, clean_img):
        """添加合成噪声"""
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_level, clean_img.shape).astype(np.float32)
            noisy = np.clip(clean_img + noise, 0, 255).astype(np.uint8)
        elif self.noise_type == "poisson":
            vals = len(np.unique(clean_img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(clean_img * vals) / float(vals)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        elif self.noise_type == "speckle":
            noise = np.random.randn(*clean_img.shape).astype(np.float32)
            noisy = np.clip(clean_img + clean_img * noise * 0.1, 0, 255).astype(np.uint8)
        else:
            noisy = clean_img
        return noisy
    
    def __getitem__(self, idx):
        clean_path = self.clean_images[idx]
        clean = cv2.imread(str(clean_path))
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        clean = cv2.resize(clean, (self.img_size, self.img_size))
        
        if self.paired:
            noisy_path = self.data_dir / "noisy" / clean_path.name
            noisy = cv2.imread(str(noisy_path))
            noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            noisy = cv2.resize(noisy, (self.img_size, self.img_size))
        else:
            noisy = self.add_noise(clean)
        
        # 归一化到 [0, 1]
        clean = torch.from_numpy(clean.transpose(2, 0, 1)).float() / 255.0
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1)).float() / 255.0
        
        return {"noisy": noisy, "clean": clean, "path": str(clean_path)}


def calculate_psnr(img1, img2):
    """计算 PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    
    for batch in dataloader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        
        optimizer.zero_grad()
        pred = model(noisy)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            total_psnr += calculate_psnr(pred, clean).item()
    
    return total_loss / len(dataloader), total_psnr / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    
    with torch.no_grad():
        for batch in dataloader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            total_loss += loss.item()
            total_psnr += calculate_psnr(pred, clean).item()
    
    return total_loss / len(dataloader), total_psnr / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="训练去噪模型")
    parser.add_argument("--data_dir", required=True, help="数据集目录")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--noise_type", default="gaussian", choices=["gaussian", "poisson", "speckle", "real"])
    parser.add_argument("--noise_level", type=int, default=25)
    parser.add_argument("--save_dir", default="./checkpoints/denoiser")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 数据集
    train_ds = DenoiseDataset(args.data_dir, "train", args.img_size, args.noise_type, args.noise_level)
    val_ds = DenoiseDataset(args.data_dir, "val", args.img_size, args.noise_type, args.noise_level)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型
    model = NAFNetMobile(img_channels=3, width=32).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    criterion = nn.L1Loss()  # L1 loss 对去噪更稳定
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    writer = SummaryWriter(log_dir=f"{args.save_dir}/runs")
    best_psnr = 0
    
    logger.info("=" * 50)
    logger.info("Denoiser training started")
    logger.info("=" * 50)
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - t0
        logger.info(f"Epoch {epoch}/{args.epochs} | "
                   f"Train Loss: {train_loss:.4f} PSNR: {train_psnr:.2f} | "
                   f"Val Loss: {val_loss:.4f} PSNR: {val_psnr:.2f} | "
                   f"Time: {epoch_time:.1f}s")
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("PSNR", {"train": train_psnr, "val": val_psnr}, epoch)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "psnr": val_psnr,
            }, f"{args.save_dir}/nafnet_best.pth")
            logger.info(f"  -> Best model saved! PSNR: {best_psnr:.2f}")
        
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
            }, f"{args.save_dir}/nafnet_epoch_{epoch}.pth")
    
    writer.close()
    logger.info("=" * 50)
    logger.info(f"Training complete! Best PSNR: {best_psnr:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
