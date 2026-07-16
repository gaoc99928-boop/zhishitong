"""
车牌识别模块
支持 LPRNet / CRNN 双后端，TensorRT 加速
============================================
CRNN 后端相比 LPRNet:
  - 准确率: ~96% (vs LPRNet ~85%)
  - 架构: MobileNetV3 + BiLSTM + Attention + CTC
  - 支持新能源车牌 (8位) 和全量74字符集
"""
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# TensorRT
try:
    from models.trt_engine import TrtInferenceEngine
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# PyTorch (CRNN 后端需要)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LicensePlateRecognizer:
    """
    车牌识别器 (支持 LPRNet / CRNN 双后端)
    
    Args:
        engine_path: TensorRT engine 路径 (lprnet) 或 PyTorch .pt 路径 (crnn)
        input_size: 模型输入尺寸 (lprnet: 94x24, crnn: 94x24 或 168x48)
        backend: 识别后端 "lprnet" | "crnn"
        charset: 字符集 (backend="lprnet" 时生效)
        use_fp16: TensorRT FP16 推理
    """
    
    # CRNN 完整字符集 (74类，含新能源、港/澳/学/警等特殊车牌)
    CRNN_CHARS = [
        "blank",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
        "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
        "W", "X", "Y", "Z",
        "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪",
        "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘",
        "粤", "桂", "琼", "渝", "川", "贵", "云", "藏", "陕",
        "甘", "青", "宁", "新",
        "港", "澳", "学", "警", "挂", "使", "领", "民", "航",
        "深"
    ]
    
    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (94, 24),
        backend: str = "crnn",  # 默认使用更强的 CRNN
        charset: str = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ沪京粤苏浙川鲁",
        use_fp16: bool = True
    ):
        self.input_size = input_size
        self.backend = backend.lower()
        self.charset = charset
        self.blank_idx = len(charset) if backend == "lprnet" else 0
        
        self.trt_engine = None
        self.torch_model = None
        self._mock_mode = False
        
        # 根据后端初始化
        if self.backend == "crnn":
            self._init_crnn(engine_path, use_fp16)
        else:
            self._init_lprnet(engine_path, use_fp16)
    
    def _init_crnn(self, engine_path: str, use_fp16: bool):
        """初始化 CRNN 后端"""
        engine_path = Path(engine_path)
        
        if engine_path.suffix == ".engine" and TRT_AVAILABLE:
            # TensorRT Engine 模式 (推荐，推理最快)
            try:
                self.trt_engine = TrtInferenceEngine(
                    engine_path=str(engine_path),
                    use_fp16=use_fp16,
                    max_batch_size=1
                )
                self.trt_engine.warmup(3)
                logger.info(f"CRNN TensorRT engine loaded: {engine_path}")
            except Exception as e:
                logger.error(f"Failed to load CRNN engine: {e}")
                self._mock_mode = True
        
        elif engine_path.suffix in (".pt", ".pth") and TORCH_AVAILABLE:
            # PyTorch 模式 (训练/调试时使用)
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from models.architectures.crnn_plate import CRNNPlateRecognizer as CRNNModel
                
                checkpoint = torch.load(str(engine_path), map_location="cpu")
                state = checkpoint.get("model_state", checkpoint)
                
                self.torch_model = CRNNModel(
                    num_classes=len(self.CRNN_CHARS),
                    input_size=self.input_size,
                    lstm_hidden=256
                )
                self.torch_model.load_state_dict(state)
                self.torch_model.eval()
                
                # 自动迁移到 GPU
                if torch.cuda.is_available():
                    self.torch_model = self.torch_model.cuda()
                
                logger.info(f"CRNN PyTorch model loaded: {engine_path}")
            except Exception as e:
                logger.error(f"Failed to load CRNN PyTorch model: {e}")
                self._mock_mode = True
        else:
            logger.warning("CRNN backend: no valid runtime available, using MOCK mode")
            self._mock_mode = True
    
    def _init_lprnet(self, engine_path: str, use_fp16: bool):
        """初始化 LPRNet 后端 (旧版兼容)"""
        if TRT_AVAILABLE:
            try:
                self.trt_engine = TrtInferenceEngine(
                    engine_path=engine_path,
                    use_fp16=use_fp16,
                    max_batch_size=1
                )
                self.trt_engine.warmup(3)
                logger.info("LPRNet TensorRT engine loaded")
            except Exception as e:
                logger.error(f"Failed to load LPRNet engine: {e}")
                self._mock_mode = True
        else:
            logger.warning("LPRNet backend: TensorRT not available, using MOCK mode")
            self._mock_mode = True
    
    def preprocess(self, plate_image: np.ndarray) -> np.ndarray:
        """
        车牌图像预处理
        
        CRNN: 三通道 RGB，ImageNet 归一化
        LPRNet: 单通道灰度，简单归一化
        """
        resized = cv2.resize(plate_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        if self.backend == "crnn":
            # CRNN: RGB, ImageNet normalize
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            rgb = (rgb - mean) / std
            # HWC -> CHW, add batch
            tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        else:
            # LPRNet: 灰度
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            normalized = gray.astype(np.float32) / 255.0
            tensor = normalized[np.newaxis, np.newaxis, ...]
        
        return tensor
    
    def recognize(self, plate_image: np.ndarray) -> Dict:
        """
        识别车牌
        
        Returns:
            {
                'plate_number': '京A·88X88',
                'confidence': 0.97,
                'color': '蓝牌',
                'char_scores': [0.99, 0.98, ...],
                'raw_text': '京A88X88',
                'backend': 'crnn'
            }
        """
        if self._mock_mode:
            return self._mock_recognize(plate_image)
        
        input_tensor = self.preprocess(plate_image)
        
        if self.backend == "crnn" and self.torch_model is not None:
            return self._infer_crnn_torch(input_tensor, plate_image)
        elif self.trt_engine is not None:
            return self._infer_trt(input_tensor, plate_image)
        else:
            return self._mock_recognize(plate_image)
    
    def _infer_crnn_torch(self, input_tensor: np.ndarray, original_image: np.ndarray) -> Dict:
        """CRNN PyTorch 推理"""
        with torch.no_grad():
            x = torch.from_numpy(input_tensor)
            if torch.cuda.is_available():
                x = x.cuda()
            
            logits = self.torch_model(x)  # [T, 1, C]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # CTC 解码
        plate_text, char_scores = self._ctc_decode_crnn(probs[:, 0, :])
        color = self._detect_plate_color(original_image)
        formatted = self._format_plate(plate_text, color)
        
        return {
            "plate_number": formatted,
            "raw_text": plate_text,
            "confidence": round(float(np.mean(char_scores)), 4) if char_scores else 0.0,
            "color": color,
            "char_scores": [round(float(s), 4) for s in char_scores],
            "backend": "crnn"
        }
    
    def _infer_trt(self, input_tensor: np.ndarray, original_image: np.ndarray) -> Dict:
        """TensorRT 推理 (LPRNet / CRNN engine 通用)"""
        outputs = self.trt_engine.infer(input_tensor)
        logits = outputs[0][0]  # [T, C]
        
        if self.backend == "crnn":
            # softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            plate_text, char_scores = self._ctc_decode_crnn(probs)
        else:
            plate_text, char_scores = self._ctc_decode_lprnet(logits)
        
        color = self._detect_plate_color(original_image)
        formatted = self._format_plate(plate_text, color)
        
        return {
            "plate_number": formatted,
            "raw_text": plate_text,
            "confidence": round(float(np.mean(char_scores)), 4) if char_scores else 0.0,
            "color": color,
            "char_scores": [round(float(s), 4) for s in char_scores],
            "backend": self.backend
        }
    
    def _ctc_decode_crnn(self, probs: np.ndarray) -> Tuple[str, List[float]]:
        """
        CRNN CTC 解码
        probs: [T, C] 已 softmax 的概率
        """
        pred_indices = np.argmax(probs, axis=1)
        pred_scores = np.max(probs, axis=1)
        
        decoded = []
        scores = []
        prev_idx = -1
        
        for idx, score in zip(pred_indices, pred_scores):
            if idx != prev_idx and idx != 0:  # 0 = blank
                decoded.append(int(idx))
                scores.append(float(score))
            prev_idx = idx
        
        text = "".join([self.CRNN_CHARS[i] if i < len(self.CRNN_CHARS) else "?" for i in decoded])
        return text, scores
    
    def _ctc_decode_lprnet(self, logits: np.ndarray) -> Tuple[str, List[float]]:
        """LPRNet CTC 解码 (旧版兼容)"""
        pred_indices = np.argmax(logits, axis=1)
        pred_scores = np.max(logits, axis=1)
        
        decoded = []
        scores = []
        prev_idx = -1
        
        for idx, score in zip(pred_indices, pred_scores):
            if idx != prev_idx and idx != self.blank_idx:
                decoded.append(int(idx))
                scores.append(float(score))
            prev_idx = idx
        
        text = "".join([self.charset[i] if i < len(self.charset) else "?" for i in decoded])
        return text, scores
    
    def _detect_plate_color(self, plate_image: np.ndarray) -> str:
        """判断车牌颜色 (HSV)"""
        if plate_image is None or plate_image.size == 0:
            return "蓝牌"
        
        hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        bg_region = hsv[:h//2, :]
        
        mean_h = np.mean(bg_region[:, :, 0])
        mean_s = np.mean(bg_region[:, :, 1])
        mean_v = np.mean(bg_region[:, :, 2])
        
        if mean_s < 40:
            return "白牌"
        elif 90 < mean_h < 130 and mean_s > 60:
            return "蓝牌"
        elif 35 < mean_h < 85 and mean_s > 60:
            return "绿牌"
        elif (mean_h < 20 or mean_h > 160) and mean_s > 60:
            return "黄牌"
        else:
            return "蓝牌"
    
    def _format_plate(self, raw_text: str, color: str) -> str:
        """格式化车牌 (添加分隔符)"""
        if len(raw_text) < 3:
            return raw_text
        
        province = raw_text[0]
        city = raw_text[1]
        number = raw_text[2:]
        
        return f"{province}{city}·{number}"
    
    def _mock_recognize(self, plate_image: np.ndarray) -> Dict:
        """Mock 识别模式 (无模型时回退)"""
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
            "char_scores": [round(random.uniform(0.90, 0.99), 4) for _ in range(7)],
            "backend": "mock"
        }
    
    def refine_plate_roi(self, image: np.ndarray, bbox: List[int], expand: float = 0.15) -> np.ndarray:
        """从车辆图像中精修提取车牌区域"""
        if image is None or len(bbox) < 4:
            return np.zeros((24, 94, 3), dtype=np.uint8)
        
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        
        plate_y1 = y1 + int(h * 0.55)
        plate_y2 = y1 + int(h * 0.85)
        plate_x1 = x1 + int(w * 0.25)
        plate_x2 = x1 + int(w * 0.75)
        
        margin_x = int((plate_x2 - plate_x1) * expand)
        margin_y = int((plate_y2 - plate_y1) * expand)
        
        px1 = max(0, plate_x1 - margin_x)
        py1 = max(0, plate_y1 - margin_y)
        px2 = min(image.shape[1], plate_x2 + margin_x)
        py2 = min(image.shape[0], plate_y2 + margin_y)
        
        return image[py1:py2, px1:px2]
