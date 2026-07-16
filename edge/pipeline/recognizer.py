"""
车牌识别模块
基于 LPRNet + TensorRT
支持中文字符集、蓝牌/绿牌/黄牌
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from models.trt_engine import TrtInferenceEngine
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class LicensePlateRecognizer:
    """
    车牌识别器
    
    功能:
    - 车牌字符识别 (LPRNet)
    - 车牌颜色判断 (蓝/绿/黄)
    - 车牌区域精修
    - 置信度评估
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (94, 24),
        charset: str = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ沪京粤苏浙川鲁",
        use_fp16: bool = True
    ):
        self.input_size = input_size
        self.charset = charset
        self.blank_idx = len(charset)
        
        self.trt_engine = None
        if TRT_AVAILABLE:
            try:
                self.trt_engine = TrtInferenceEngine(
                    engine_path=engine_path,
                    use_fp16=use_fp16,
                    max_batch_size=1
                )
                self.trt_engine.warmup(3)
                logger.info("Recognizer: LPRNet TensorRT engine loaded")
            except Exception as e:
                logger.error(f"Failed to load recognizer engine: {e}")
                raise
        else:
            logger.warning("Recognizer: running in MOCK mode")
            self._mock_mode = True
    
    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        车牌图像预处理
        
        Args:
            plate_image: 车牌区域图像 (BGR)
            
        Returns:
            [1, C, H, W] 归一化张量
        """
        # 缩放至模型输入尺寸
        resized = cv2.resize(plate_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # 灰度化或保留 RGB
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # 归一化到 [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        # 添加通道维度 -> [1, H, W]
        tensor = normalized[np.newaxis, np.newaxis, ...]
        
        return tensor
    
    def recognize(self, plate_image: np.ndarray) -> Dict:
        """
        识别车牌
        
        Args:
            plate_image: 车牌区域图像 (BGR)
            
        Returns:
            {
                'plate_number': '京A·88X88',
                'confidence': 0.97,
                'color': '蓝牌',
                'char_scores': [0.99, 0.98, ...],
                'raw_text': '京A88X88'
            }
        """
        if not TRT_AVAILABLE or getattr(self, '_mock_mode', False):
            return self._mock_recognize(plate_image)
        
        # 预处理
        input_tensor = self.preprocess(plate_image)
        
        # 推理
        outputs = self.trt_engine.infer(input_tensor)
        
        # CTC 解码
        logits = outputs[0][0]  # [T, C]
        plate_text, char_scores = self._ctc_decode(logits)
        
        # 车牌颜色判断
        color = self._detect_plate_color(plate_image)
        
        # 格式化 (添加分隔符)
        formatted = self._format_plate(plate_text, color)
        
        avg_conf = float(np.mean(char_scores)) if char_scores else 0.0
        
        return {
            "plate_number": formatted,
            "raw_text": plate_text,
            "confidence": round(avg_conf, 4),
            "color": color,
            "char_scores": [round(s, 4) for s in char_scores]
        }
    
    def _ctc_decode(self, logits: np.ndarray) -> Tuple[str, List[float]]:
        """
        CTC 贪心解码
        
        Args:
            logits: [T, C] 时间步 x 字符类别
            
        Returns:
            (解码字符串, 每个字符的置信度)
        """
        # 取每个时间步的最大概率字符
        pred_indices = np.argmax(logits, axis=1)
        pred_scores = np.max(logits, axis=1)
        
        # 移除重复和 blank
        decoded = []
        scores = []
        prev_idx = -1
        
        for idx, score in zip(pred_indices, pred_scores):
            if idx != prev_idx and idx != self.blank_idx:
                decoded.append(int(idx))
                scores.append(float(score))
            prev_idx = idx
        
        # 索引转字符
        text = "".join([self.charset[i] if i < len(self.charset) else "?" for i in decoded])
        
        return text, scores
    
    def _detect_plate_color(self, plate_image: np.ndarray) -> str:
        """
        判断车牌颜色
        
        基于 HSV 颜色空间分析背景色
        """
        hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        
        # 取图像上半部分作为背景参考 (车牌文字通常在下半)
        h, w = hsv.shape[:2]
        bg_region = hsv[:h//2, :]
        
        # 计算平均 HSV
        mean_h = np.mean(bg_region[:, :, 0])
        mean_s = np.mean(bg_region[:, :, 1])
        mean_v = np.mean(bg_region[:, :, 2])
        
        # 判断逻辑 (OpenCV HSV 范围: H[0-179], S[0-255], V[0-255])
        if mean_s < 40:
            return "白牌"
        elif 90 < mean_h < 130 and mean_s > 60:
            return "蓝牌"
        elif 35 < mean_h < 85 and mean_s > 60:
            return "绿牌"
        elif (mean_h < 20 or mean_h > 160) and mean_s > 60:
            return "黄牌"
        else:
            return "蓝牌"  # 默认
    
    def _format_plate(self, raw_text: str, color: str) -> str:
        """
        格式化车牌号码 (添加省份分隔符)
        
        例: 京A88X88 -> 京A·88X88
        """
        if len(raw_text) < 3:
            return raw_text
        
        # 中国车牌格式: 省份(1汉字) + 城市代码(1字母) + 编号(5-6位)
        province = raw_text[0]
        city = raw_text[1]
        number = raw_text[2:]
        
        return f"{province}{city}·{number}"
    
    def _mock_recognize(self, plate_image: np.ndarray) -> Dict:
        """Mock 识别模式"""
        import random
        provinces = ["京", "沪", "粤", "苏", "浙", "川", "鲁"]
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        nums = "0123456789"
        
        p = random.choice(provinces)
        c = random.choice(chars)
        n = "".join([random.choice(nums + chars) for _ in range(5)])
        raw = f"{p}{c}{n}"
        
        return {
            "plate_number": f"{p}{c}·{n}",
            "raw_text": raw,
            "confidence": round(random.uniform(0.88, 0.99), 4),
            "color": random.choice(["蓝牌", "绿牌", "黄牌"]),
            "char_scores": [round(random.uniform(0.90, 0.99), 4) for _ in range(7)]
        }
    
    def refine_plate_roi(self, image: np.ndarray, bbox: List[int], expand: float = 0.15) -> np.ndarray:
        """
        从车辆图像中精修提取车牌区域
        
        基于先验知识: 车牌通常在车辆下半部分
        """
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        
        # 车牌通常在车辆下半部中央
        plate_y1 = y1 + int(h * 0.55)
        plate_y2 = y1 + int(h * 0.85)
        plate_x1 = x1 + int(w * 0.25)
        plate_x2 = x1 + int(w * 0.75)
        
        # 扩展
        margin_x = int((plate_x2 - plate_x1) * expand)
        margin_y = int((plate_y2 - plate_y1) * expand)
        
        px1 = max(0, plate_x1 - margin_x)
        py1 = max(0, plate_y1 - margin_y)
        px2 = min(image.shape[1], plate_x2 + margin_x)
        py2 = min(image.shape[0], plate_y2 + margin_y)
        
        return image[py1:py2, px1:px2]
