"""
图像处理工具函数
预处理、后处理、可视化辅助
"""
import cv2
import numpy as np
from typing import Tuple, List


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    保持长宽比的图像缩放 (Letterbox)
    
    Returns:
        (padded_image, scale, padding)
    """
    h, w = image.shape[:2]
    th, tw = target_size
    
    scale = min(th / h, tw / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_h = (th - new_h) // 2
    pad_w = (tw - new_w) // 2
    
    padded = np.full((th, tw, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return padded, scale, (pad_w, pad_h)


def restore_bbox(
    bbox: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int]
) -> np.ndarray:
    """
    将 Letterbox 后的 bbox 还原到原图坐标
    """
    pad_w, pad_h = padding
    bbox = bbox.copy()
    bbox[:, [0, 2]] -= pad_w
    bbox[:, [1, 3]] -= pad_h
    bbox /= scale
    bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, orig_shape[1])
    bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, orig_shape[0])
    return bbox


def draw_bbox(
    image: np.ndarray,
    bbox: List[int],
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """绘制检测框和标签"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        
        text_x = x1
        text_y = y1 - 5 if y1 > 20 else y1 + text_size[1] + 5
        
        cv2.rectangle(
            image,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y),
            color,
            -1
        )
        cv2.putText(image, label, (text_x, text_y - 2), font, font_scale, (0, 0, 0), 2)
    
    return image


def create_comparison_view(
    original: np.ndarray,
    enhanced: np.ndarray,
    labels: Tuple[str, str] = ("Original", "Enhanced")
) -> np.ndarray:
    """创建原图/增强图对比视图"""
    h, w = original.shape[:2]
    
    # 创建横向拼接
    separator = np.full((h, 10, 3), (64, 64, 64), dtype=np.uint8)
    combined = np.hstack([original, separator, enhanced])
    
    # 添加标签
    cv2.putText(combined, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, labels[1], (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return combined


def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    基于中值绝对偏差估计高斯噪声标准差
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    med = cv2.medianBlur(gray, 5)
    diff = np.abs(gray.astype(np.float32) - med.astype(np.float32))
    mad = np.median(diff)
    return mad * 1.4826


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """应用 CLAHE 自适应直方图均衡"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Gamma 校正"""
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)
