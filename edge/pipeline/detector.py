"""
车辆检测模块
基于 YOLOv8-Nano + TensorRT
支持多类别车辆检测与跟踪
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


class VehicleDetector:
    """
    车辆检测器
    
    功能:
    - 多类别车辆检测 (car, suv, truck, bus, van, taxi)
    - NMS 后处理
    - 检测框质量评估
    - 与 DeepStream tracker 集成接口
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.50,
        labels: List[str] = None,
        use_fp16: bool = True
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.labels = labels or ["car", "suv", "truck", "bus", "van", "taxi"]
        
        self.trt_engine = None
        if TRT_AVAILABLE:
            try:
                self.trt_engine = TrtInferenceEngine(
                    engine_path=engine_path,
                    use_fp16=use_fp16,
                    max_batch_size=1
                )
                self.trt_engine.warmup(3)
                logger.info("Detector: YOLOv8 TensorRT engine loaded")
            except Exception as e:
                logger.error(f"Failed to load detector engine: {e}")
                raise
        else:
            logger.warning("Detector: running in MOCK mode")
            self._mock_mode = True
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        图像预处理
        
        Returns:
            (tensor, scale_factors, padding)
        """
        h, w = image.shape[:2]
        
        # 计算缩放和填充 (letterbox)
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充画布
        pad_h = (self.input_size[0] - new_h) // 2
        pad_w = (self.input_size[1] - new_w) // 2
        
        padded = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        
        return tensor, (scale, scale), (pad_w, pad_h)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        执行车辆检测
        
        Args:
            image: BGR 格式输入图像
            
        Returns:
            检测结果列表，每项包含:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str,
                'area': int
            }
        """
        if not TRT_AVAILABLE or getattr(self, '_mock_mode', False):
            return self._mock_detect(image)
        
        # 预处理
        input_tensor, scale, padding = self.preprocess(image)
        
        # 推理
        outputs = self.trt_engine.infer(input_tensor)
        
        # 后处理
        detections = self._postprocess(outputs[0], scale, padding, image.shape[:2])
        
        return detections
    
    def _postprocess(
        self,
        output: np.ndarray,
        scale: Tuple[float, float],
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        YOLOv8 输出后处理
        
        Args:
            output: [1, 84, 8400] 或 [1, 8400, 84] 原始输出
        """
        # 适配不同输出格式
        if output.shape[1] == 84:
            predictions = output[0].T  # [8400, 84]
        else:
            predictions = output[0]  # [8400, 84]
        
        # 过滤低置信度
        scores = np.max(predictions[:, 4:], axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(predictions) == 0:
            return []
        
        # 解码 bbox
        boxes = predictions[:, :4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # 转换 xywh -> xyxy
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 反归一化到输入尺寸
        xyxy[:, [0, 2]] *= self.input_size[1]
        xyxy[:, [1, 3]] *= self.input_size[0]
        
        # 移除填充并缩放到原图
        pad_w, pad_h = padding
        xyxy[:, [0, 2]] -= pad_w
        xyxy[:, [1, 3]] -= pad_h
        xyxy /= scale[0]
        
        # 裁剪到图像边界
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_shape[1])
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_shape[0])
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # 整理结果
        results = []
        for idx in indices.flatten():
            x1, y1, x2, y2 = xyxy[idx].astype(int)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(scores[idx]), 4),
                "class_id": int(class_ids[idx]),
                "class_name": self.labels[int(class_ids[idx])] if int(class_ids[idx]) < len(self.labels) else "unknown",
                "area": int((x2 - x1) * (y2 - y1))
            })
        
        # 按置信度排序
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict]:
        """Mock 检测模式，返回模拟结果"""
        h, w = image.shape[:2]
        return [
            {
                "bbox": [int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.8)],
                "confidence": 0.92,
                "class_id": 0,
                "class_name": "car",
                "area": int(w*0.4 * h*0.4)
            }
        ]
    
    def crop_vehicle(self, image: np.ndarray, det: Dict, expand_ratio: float = 0.0) -> np.ndarray:
        """
        根据检测框裁剪车辆区域
        
        Args:
            expand_ratio: 扩展比例，用于包含更多上下文
        """
        x1, y1, x2, y2 = det["bbox"]
        h, w = image.shape[:2]
        
        if expand_ratio > 0:
            margin_x = int((x2 - x1) * expand_ratio)
            margin_y = int((y2 - y1) * expand_ratio)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
        
        return image[y1:y2, x1:x2]
    
    def estimate_scene(self, image: np.ndarray, detections: List[Dict]) -> str:
        """
        简单场景估计 (正常/雨雾/夜间/逆光)
        基于全局亮度和对比度
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        
        # 检测车灯 (高亮区域)
        bright_pixels = np.sum(gray > 200)
        bright_ratio = bright_pixels / gray.size
        
        if mean_brightness < 50:
            return "night"
        elif std_contrast < 30 and mean_brightness > 150:
            return "backlight"
        elif std_contrast < 25 and mean_brightness < 100:
            return "rain"
        else:
            return "normal"
