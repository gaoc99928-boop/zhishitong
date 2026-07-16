"""
车牌识别模型训练脚本 (CRNN + CTC)
====================================
支持从 CCPD 数据集训练，产出 .pt 权重和 .onnx 导出文件

使用方法:
    # 完整训练 (CCPD 数据集)
    python train_plate_recognition.py --data_dir /path/to/ccpd --epochs 50
    
    # 快速验证 (小批量)
    python train_plate_recognition.py --data_dir /path/to/ccpd --epochs 2 --batch_size 16
    
    # 继续训练
    python train_plate_recognition.py --resume checkpoints/crnn_epoch_20.pt

预期性能:
    - CCPD-Base 测试集准确率: ~96%+
    - 单帧推理延迟 (Jetson Orin Nano FP16): ~8ms
"""
import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch is required for training. Install: pip install torch torchvision")
    sys.exit(1)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.architectures.crnn_plate import CRNNPlateRecognizer, get_char_mapping
from train.dataset_ccpd import CCPDDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class CTCLossWrapper(nn.Module):
    """CTC Loss 包装，处理变长序列"""
    def __init__(self, blank=0, reduction="mean"):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
    
    def forward(self, logits, targets, target_lengths, input_lengths):
        # logits: [T, B, C]
        # targets: [sum(label_lens)]
        # target_lengths: [B]
        # input_lengths: [B]
        return self.ctc(logits, targets, input_lengths, target_lengths)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lens = batch["label_lens"].to(device)
        plate_texts = batch["plate_texts"]
        
        optimizer.zero_grad()
        
        # 前向传播
        logits = model(images)  # [T, B, C]
        T, B, C = logits.shape
        
        # CTC 输入长度 (时间步)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        
        # 计算损失
        loss = criterion(logits, labels, label_lens, input_lengths)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率 (贪心解码)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1).cpu().numpy()  # [T, B]
            for b in range(B):
                pred_text = CCPDDataset.decode(preds[:, b])
                if pred_text == plate_texts[b]:
                    total_correct += 1
                total_samples += 1
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                       f"Loss: {loss.item():.4f} Acc: {total_correct/max(1,total_samples):.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lens = batch["label_lens"].to(device)
            plate_texts = batch["plate_texts"]
            
            logits = model(images)
            T, B, C = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            
            loss = criterion(logits, labels, label_lens, input_lengths)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for b in range(B):
                pred_text = CCPDDataset.decode(preds[:, b])
                if pred_text == plate_texts[b]:
                    total_correct += 1
                total_samples += 1
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc


def export_onnx(model, save_path, input_size=(1, 3, 24, 94)):
    """导出 ONNX 模型"""
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    logger.info(f"ONNX exported: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="训练车牌识别模型")
    parser.add_argument("--data_dir", required=True, help="CCPD 数据集根目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_w", type=int, default=94, help="输入图像宽度")
    parser.add_argument("--img_h", type=int, default=24, help="输入图像高度")
    parser.add_argument("--lstm_hidden", type=int, default=256)
    parser.add_argument("--save_dir", default="./checkpoints", help="模型保存目录")
    parser.add_argument("--resume", default=None, help="恢复训练的检查点路径")
    parser.add_argument("--device", default="cuda", help="训练设备")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 数据集
    logger.info("Loading datasets...")
    train_loader = CCPDDataset.get_dataloader(
        args.data_dir, "train", args.batch_size, args.num_workers,
        img_size=(args.img_w, args.img_h), augment=True
    )
    val_loader = CCPDDataset.get_dataloader(
        args.data_dir, "val", args.batch_size, args.num_workers,
        img_size=(args.img_w, args.img_h), augment=False
    )
    
    # 模型
    model = CRNNPlateRecognizer(
        num_classes=CCPDDataset.NUM_CLASSES,
        input_size=(args.img_w, args.img_h),
        lstm_hidden=args.lstm_hidden
    ).to(device)
    
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 损失函数和优化器
    criterion = CTCLossWrapper(blank=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    best_acc = 0.0
    
    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch-1}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"{args.save_dir}/runs")
    
    # 训练循环
    logger.info("=" * 50)
    logger.info("Training started")
    logger.info("=" * 50)
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        
        logger.info(f"Epoch {epoch}/{args.epochs} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                   f"LR: {lr:.6f} | Time: {epoch_time:.1f}s")
        
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", lr, epoch)
        
        # 保存检查点
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "val_acc": val_acc,
        }
        
        torch.save(checkpoint, f"{args.save_dir}/crnn_last.pt")
        if is_best:
            torch.save(checkpoint, f"{args.save_dir}/crnn_best.pt")
            logger.info(f"  -> Best model saved! Acc: {best_acc:.4f}")
        
        # 每 10 个 epoch 导出 ONNX
        if epoch % 10 == 0:
            export_onnx(model, f"{args.save_dir}/crnn_epoch_{epoch}.onnx",
                       input_size=(1, 3, args.img_h, args.img_w))
    
    # 最终导出
    export_onnx(model, f"{args.save_dir}/crnn_final.onnx",
               input_size=(1, 3, args.img_h, args.img_w))
    
    writer.close()
    logger.info("=" * 50)
    logger.info(f"Training complete! Best val accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {args.save_dir}/crnn_best.pt")
    logger.info("Next step: convert to TensorRT with tools/convert_lprnet_trt.py")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
