"""
车辆类型分类器 (Vehicle Type Classifier)
============================================
基于 ResNet18/ShuffleNetV2 的轻量车型分类，TensorRT 加速
支持 7 类车型: 小型客车, SUV, 大型货车, 中型客车, 面包车, 轿车, 出租车
"""
import logging
from typing import Dict
import numpy as np
import cv2

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from models.trt_engine import TrtInferenceEngine

logger = logging.getLogger(__name__)


class VehicleClassifier:
    """
    车辆类型分类器
    
    使用说明:
        classifier = VehicleClassifier(
            engine_path="./models/engines/vehicle_type_resnet18.engine",
            input_size=(224, 224),
            labels=["小型客车", "SUV", "大型货车", "中型客车", "面包车", "轿车", "出租车"]
        )
        result = classifier.classify(vehicle_crop)
    """
    
    def __init__(
        self,
        engine_path: str,
        input_size: tuple = (224, 224),
        labels: list = None
    ):
        self.engine_path = engine_path
        self.input_size = input_size
        self.labels = labels or ["小型客车", "SUV", "大型货车", "中型客车", "面包车", "轿车", "出租车"]
        
        # 初始化 TensorRT 引擎
        if TRT_AVAILABLE and engine_path:
            try:
                self.engine = TrtInferenceEngine(engine_path)
                logger.info(f"Classifier engine loaded: {engine_path}")
            except Exception as e:
                logger.warning(f"Failed to load classifier engine: {e}, using mock mode")
                self.engine = None
        else:
            self.engine = None
            logger.info("Classifier running in mock mode (no TensorRT)")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        预处理车辆图像
        
        Steps:
            1. Resize to input_size
            2. BGR -> RGB
            3. Normalize with ImageNet stats
            4. HWC -> CHW
        """
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # CHW format
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)  # Add batch dimension
    
    def classify(self, vehicle_image: np.ndarray) -> Dict:
        """
        对车辆图像进行类型分类
        
        Args:
            vehicle_image: 车辆区域的 BGR 图像
            
        Returns:
            {
                'class_name': str,      # 车型名称
                'class_id': int,        # 类别索引
                'confidence': float,    # 置信度 (0-1)
                'all_scores': dict      # 所有类别的分数
            }
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return {
                "class_name": "unknown",
                "class_id": -1,
                "confidence": 0.0,
                "all_scores": {}
            }
        
        input_batch = self.preprocess(vehicle_image)
        
        if self.engine:
            outputs = self.engine.infer(input_batch)
            scores = outputs[0].flatten()
        else:
            # Mock mode: random softmax
            scores = np.random.rand(len(self.labels))
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        all_scores = {
            label: float(prob)
            for label, prob in zip(self.labels, probs)
        }
        
        return {
            "class_name": self.labels[class_id] if class_id < len(self.labels) else "unknown",
            "class_id": class_id,
            "confidence": confidence,
            "all_scores": all_scores
        }
