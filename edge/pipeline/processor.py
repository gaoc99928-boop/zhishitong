"""
图像处理流水线 Orchestrator
整合去噪、检测、识别、分类等模块
提供完整的单帧处理接口
"""
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from pipeline.denoiser import ImageDenoiser
from pipeline.detector import VehicleDetector
from pipeline.recognizer import LicensePlateRecognizer

logger = logging.getLogger(__name__)


class VehiclePipeline:
    """
    车辆图像处理流水线
    
    处理流程:
    1. 场景评估 (亮度/对比度/噪声)
    2. 按需去噪增强 (如果图像质量差)
    3. 车辆检测 (YOLOv8)
    4. 车牌定位与识别 (LPRNet)
    5. 车型分类 (ResNet18)
    6. 结构化输出
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        self.config = self._load_config(config_path)
        self.models_config = self.config.get("models", {})
        self.pipeline_config = self.config.get("pipeline", {})
        
        # 初始化各模块
        self.denoiser = None
        self.detector = None
        self.recognizer = None
        self.classifier = None
        
        self._init_modules()
        
        # 性能统计
        self.stats = {
            "total_frames": 0,
            "total_vehicles": 0,
            "avg_latency_ms": 0.0,
            "denoise_count": 0
        }
    
    def _load_config(self, path: str) -> dict:
        """加载 YAML 配置"""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _init_modules(self):
        """初始化各处理模块"""
        # 去噪模块
        denoiser_cfg = self.models_config.get("denoiser", {})
        if denoiser_cfg:
            self.denoiser = ImageDenoiser(
                engine_path=denoiser_cfg.get("engine_path"),
                input_size=tuple(denoiser_cfg.get("input_size", [256, 256])),
                quality_threshold=denoiser_cfg.get("quality_threshold", 65),
                mode=self.pipeline_config.get("denoise", {}).get("mode", "on_demand")
            )
        
        # 检测模块
        detector_cfg = self.models_config.get("detector", {})
        if detector_cfg:
            self.detector = VehicleDetector(
                engine_path=detector_cfg.get("engine_path"),
                input_size=tuple(detector_cfg.get("input_size", [640, 640])),
                conf_threshold=detector_cfg.get("conf_threshold", 0.45),
                nms_threshold=detector_cfg.get("nms_threshold", 0.50),
                labels=detector_cfg.get("labels")
            )
        
        # 识别模块
        recognizer_cfg = self.models_config.get("recognizer", {})
        if recognizer_cfg:
            self.recognizer = LicensePlateRecognizer(
                engine_path=recognizer_cfg.get("engine_path"),
                input_size=tuple(recognizer_cfg.get("input_size", [94, 24])),
                charset=recognizer_cfg.get("charset")
            )
        
        logger.info("Pipeline initialized successfully")
    
    def process_frame(self, image: np.ndarray, frame_id: int = 0) -> Dict:
        """
        处理单帧图像
        
        Args:
            image: BGR 格式输入图像
            frame_id: 帧编号
            
        Returns:
            {
                'frame_id': int,
                'timestamp': str,
                'scene_type': str,
                'processing_time_ms': float,
                'vehicles': [
                    {
                        'track_id': int,
                        'bbox': [x1, y1, x2, y2],
                        'vehicle_type': str,
                        'plate': {
                            'number': str,
                            'color': str,
                            'confidence': float
                        },
                        'detection_conf': float,
                        'scene_enhanced': bool
                    }
                ]
            }
        """
        t_start = time.perf_counter()
        
        result = {
            "frame_id": frame_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scene_type": "normal",
            "processing_time_ms": 0.0,
            "vehicles": []
        }
        
        if image is None or image.size == 0:
            logger.warning("Empty image received")
            return result
        
        try:
            # Step 1: 场景评估与去噪
            working_image = image.copy()
            scene_enhanced = False
            
            if self.denoiser:
                quality = self.denoiser.assess_quality(working_image)
                result["scene_type"] = self._map_scene(quality)
                
                if quality["is_noisy"]:
                    working_image, denoise_meta = self.denoiser.denoise(working_image)
                    scene_enhanced = denoise_meta.get("method") != "none"
                    if scene_enhanced:
                        self.stats["denoise_count"] += 1
            
            # Step 2: 车辆检测
            if self.detector is None:
                logger.error("Detector not initialized")
                return result
            
            detections = self.detector.detect(working_image)
            
            if not detections:
                result["processing_time_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                return result
            
            # Step 3: 对每个检测到的车辆进行识别
            for i, det in enumerate(detections):
                vehicle_info = {
                    "track_id": frame_id * 100 + i,  # 简化 track_id
                    "bbox": det["bbox"],
                    "vehicle_type": det.get("class_name", "unknown"),
                    "detection_conf": det.get("confidence", 0.0),
                    "scene_enhanced": scene_enhanced
                }
                
                # 车牌识别
                if self.recognizer:
                    plate_roi = self.recognizer.refine_plate_roi(working_image, det["bbox"])
                    if plate_roi.size > 0:
                        plate_result = self.recognizer.recognize(plate_roi)
                        vehicle_info["plate"] = {
                            "number": plate_result["plate_number"],
                            "color": plate_result["color"],
                            "confidence": plate_result["confidence"],
                            "raw_text": plate_result["raw_text"]
                        }
                    else:
                        vehicle_info["plate"] = None
                else:
                    vehicle_info["plate"] = None
                
                result["vehicles"].append(vehicle_info)
            
            # 更新统计
            self.stats["total_frames"] += 1
            self.stats["total_vehicles"] += len(detections)
            
            latency = (time.perf_counter() - t_start) * 1000
            result["processing_time_ms"] = round(latency, 2)
            
            # 更新平均延迟
            n = self.stats["total_frames"]
            self.stats["avg_latency_ms"] = (
                self.stats["avg_latency_ms"] * (n - 1) + latency
            ) / n
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}", exc_info=True)
            result["error"] = str(e)
        
        return result
    
    def _map_scene(self, quality: dict) -> str:
        """将质量评估映射到场景类型"""
        score = quality.get("score", 100)
        noise = quality.get("noise_level", 0)
        
        if score < 30:
            return "night"
        elif score < 50 and noise > 15:
            return "rain"
        elif score < 50:
            return "backlight"
        else:
            return "normal"
    
    def get_stats(self) -> dict:
        """获取处理统计信息"""
        return {
            **self.stats,
            "avg_vehicles_per_frame": (
                self.stats["total_vehicles"] / max(1, self.stats["total_frames"])
            )
        }
    
    def draw_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        在图像上绘制检测结果 (用于调试/预览)
        """
        vis = image.copy()
        
        for v in result.get("vehicles", []):
            x1, y1, x2, y2 = v["bbox"]
            
            # 绘制检测框
            color = (0, 255, 0) if not v.get("scene_enhanced") else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{v['vehicle_type']}"
            if v.get("plate"):
                label += f" | {v['plate']['number']}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 绘制延迟信息
        latency = result.get("processing_time_ms", 0)
        cv2.putText(vis, f"Latency: {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis
