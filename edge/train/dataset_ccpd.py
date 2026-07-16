"""
CCPD 数据集加载器
================
支持 CCPD2019 / CCPD2020 / CCPD-Green 等变体
自动处理车牌标注、数据增强、图像归一化
"""
import os
import re
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CCPDDataset(Dataset):
    """
    CCPD 车牌识别数据集
    
    数据格式 (CCPD 文件名即标注):
        025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        格式: 区域亮度_车牌边界框_四个角点_车牌号码_亮度_模糊度.jpg
    """
    
    # 省份简称映射
    PROVINCES = [
        "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
        "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
        "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
    ]
    
    # 字母数字 (去除 I,O 避免与 1,0 混淆)
    ALPHANUM = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    
    # 完整字符集
    CHARS = ["blank"] + list(ALPHANUM) + PROVINCES + [
        "港", "澳", "学", "警", "挂", "使", "领", "民", "航", "深"
    ]
    
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS)
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (94, 24),
        augment: bool = True,
        max_samples: int = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        
        # 收集所有图像文件
        self.samples = self._load_samples()
        if max_samples:
            self.samples = random.sample(self.samples, min(max_samples, len(self.samples)))
        
        logger.info(f"CCPD {split}: {len(self.samples)} samples loaded")
    
    def _load_samples(self) -> List[Dict]:
        """加载所有样本路径和标注"""
        samples = []
        img_dir = self.data_dir / self.split
        
        if not img_dir.exists():
            # 尝试扁平结构
            img_dir = self.data_dir
        
        for img_path in img_dir.glob("*.jpg"):
            plate_str = self._parse_filename(img_path.name)
            if plate_str:
                samples.append({
                    "image_path": str(img_path),
                    "plate_text": plate_str,
                    "label": self._encode(plate_str)
                })
        
        return samples
    
    def _parse_filename(self, filename: str) -> str:
        """
        从 CCPD 文件名解析车牌号码
        格式: ..._车牌号码_亮度_模糊度.jpg
        """
        try:
            # 去掉 .jpg 后缀
            name = filename.replace(".jpg", "")
            parts = name.split("-")
            if len(parts) < 7:
                return None
            
            # CCPD 标注格式: ...-plate_code-brightness-blur.jpg
            plate_code = parts[-3]
            plate_chars = plate_code.split("_")
            
            # 解析每个字符编码
            plate_text = ""
            for code in plate_chars:
                if not code:
                    continue
                idx = int(code)
                if idx < len(self.CHARS):
                    plate_text += self.CHARS[idx]
            
            return plate_text if len(plate_text) >= 6 else None
        except Exception:
            return None
    
    def _encode(self, text: str) -> List[int]:
        """文本转索引序列"""
        return [self.CHAR_TO_IDX.get(c, 0) for c in text if c in self.CHAR_TO_IDX]
    
    @classmethod
    def decode(cls, indices: List[int]) -> str:
        """索引序列转文本 (CTC 解码: 去重+去blank)"""
        result = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:  # 0 是 blank
                result.append(cls.IDX_TO_CHAR.get(idx, ""))
            prev = idx
        return "".join(result)
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """训练时数据增强"""
        # 随机亮度
        if random.random() < 0.5:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 随机模糊 (模拟低质量摄像头)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), 0)
        
        # 随机噪声
        if random.random() < 0.2:
            noise = np.random.normal(0, 10, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 随机对比度
        if random.random() < 0.3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(sample["image_path"])
        if image is None:
            image = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        
        # 数据增强
        if self.augment:
            image = self._augment(image)
        
        # 调整尺寸
        image = cv2.resize(image, self.img_size)
        
        # BGR -> RGB, 归一化
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        label = np.array(sample["label"], dtype=np.int64)
        label_len = len(label)
        
        return {
            "image": torch.from_numpy(image) if TORCH_AVAILABLE else image,
            "label": torch.from_numpy(label) if TORCH_AVAILABLE else label,
            "label_len": label_len,
            "plate_text": sample["plate_text"],
            "img_path": sample["image_path"]
        }


def collate_fn(batch):
    """批量数据整理 (处理变长序列)"""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.cat([b["label"] for b in batch])
    label_lens = torch.tensor([b["label_len"] for b in batch], dtype=torch.long)
    plate_texts = [b["plate_text"] for b in batch]
    
    return {
        "images": images,
        "labels": labels,
        "label_lens": label_lens,
        "plate_texts": plate_texts
    }


def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (94, 24),
    augment: bool = True
):
    """获取 DataLoader"""
    dataset = CCPDDataset(data_dir, split, img_size, augment)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train")
    )
