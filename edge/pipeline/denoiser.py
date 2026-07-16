"""
图像去噪增强模块
支持两种模式:
  1. TensorRT NAFNet-mobile (GPU 加速)
  2. 传统 BM3D/NLM 回退算法 (CPU)
"""
import logging
import numpy as np
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from models.trt_engine import TrtInferenceEngine
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


class ImageDenoiser:
    """
    图像去噪增强器
    
    策略:
    - 优先使用 TensorRT 引擎 (GPU 加速, ~50ms)
    - 引擎不可用时回退到 BM3D (CPU, ~200ms)
    - 提供图像质量评估，自动决定是否增强
    """
    
    def __init__(
        self,
        engine_path: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 256),
        use_fp16: bool = True,
        quality_threshold: int = 65,
        mode: str = "on_demand"
    ):
        self.input_size = input_size
        self.quality_threshold = quality_threshold
        self.mode = mode  # real_time, on_demand, off
        
        self.trt_engine = None
        if engine_path and TRT_AVAILABLE:
            try:
                self.trt_engine = TrtInferenceEngine(
                    engine_path=engine_path,
                    use_fp16=use_fp16,
                    max_batch_size=1
                )
                self.trt_engine.warmup(2)
                logger.info("Denoiser: TensorRT engine loaded")
            except Exception as e:
                logger.warning(f"Failed to load TRT denoiser: {e}, fallback to BM3D")
        
        if self.trt_engine is None:
            logger.info("Denoiser: using BM3D/NLM fallback")
    
    def assess_quality(self, image: np.ndarray) -> dict:
        """
        评估图像质量
        
        Returns:
            {
                'score': 0-100 质量分数,
                'is_noisy': 是否需要增强,
                'noise_level': 噪声估计值,
                'blur_metric': 模糊度
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 拉普拉斯方差作为模糊度指标
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_metric = min(100, lap_var / 10)
        
        # 噪声估计 (基于小波变换或简单标准差)
        noise_level = self._estimate_noise(gray)
        
        # 综合质量分数
        score = max(0, min(100, blur_metric - noise_level * 2))
        
        return {
            "score": round(score, 1),
            "is_noisy": score < self.quality_threshold,
            "noise_level": round(noise_level, 2),
            "blur_metric": round(blur_metric, 1)
        }
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """基于中值绝对偏差估计噪声水平"""
        med = cv2.medianBlur(gray, 5)
        diff = np.abs(gray.astype(np.float32) - med.astype(np.float32))
        mad = np.median(diff)
        # MAD 到标准差的转换因子 ~1.4826
        sigma = mad * 1.4826
        return sigma
    
    def denoise(self, image: np.ndarray, force: bool = False) -> Tuple[np.ndarray, dict]:
        """
        执行去噪增强
        
        Args:
            image: BGR 格式输入图像
            force: 强制去噪，忽略质量评估
            
        Returns:
            (enhanced_image, metadata)
        """
        if self.mode == "off" and not force:
            return image, {"method": "skipped", "reason": "denoiser disabled"}
        
        # 质量评估
        quality = self.assess_quality(image)
        
        if not force and not quality["is_noisy"]:
            return image, {"method": "none", "quality": quality}
        
        # 选择去噪方法
        if self.trt_engine is not None:
            enhanced = self._denoise_trt(image)
            method = "nafnet_trt"
        else:
            enhanced = self._denoise_bm3d(image)
            method = "bm3d"
        
        metadata = {
            "method": method,
            "quality_before": quality,
            "quality_after": self.assess_quality(enhanced)
        }
        
        return enhanced, metadata
    
    def _denoise_trt(self, image: np.ndarray) -> np.ndarray:
        """使用 TensorRT NAFNet 进行去噪"""
        h, w = image.shape[:2]
        
        # 缩放至模型输入尺寸
        resized = cv2.resize(image, self.input_size)
        
        # BGR -> RGB, HWC -> CHW, normalize to [-1, 1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
        input_tensor = input_tensor * 2.0 - 1.0
        
        # 推理
        outputs = self.trt_engine.infer(input_tensor)
        
        # 后处理
        output = outputs[0][0]  # [C, H, W]
        output = np.transpose(output, (1, 2, 0))
        output = (output + 1.0) / 2.0
        output = np.clip(output, 0, 1) * 255
        output = output.astype(np.uint8)
        
        # RGB -> BGR, 缩放回原尺寸
        bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (w, h))
    
    def _denoise_bm3d(self, image: np.ndarray) -> np.ndarray:
        """
        使用 BM3D 或 NLM 进行去噪 (CPU 回退)
        
        注: cv2 不自带 BM3D，使用 NLM + 对比度增强作为等效方案
        """
        # 估计噪声水平用于自适应参数
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_sigma = self._estimate_noise(gray)
        
        # 非局部均值去噪
        denoised = cv2.fastNlMeansDenoisingColored(
            image, None,
            h=noise_sigma * 1.5,
            hColor=noise_sigma * 1.5,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        # 轻度对比度增强 (模拟 GAN 效果)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.15, beta: float = 1.05) -> np.ndarray:
        """快速对比度增强 (用于实时预览)"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        enhanced = cv2.addWeighted(enhanced, beta, image, 1 - beta, 0)
        return enhanced
