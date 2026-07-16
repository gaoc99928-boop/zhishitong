"""
TensorRT 推理引擎封装模块
适配 Jetson Orin Nano，支持 FP16/INT8 推理
"""
import os
import time
import logging
import numpy as np
from typing import List, Tuple, Optional

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available, running in mock mode")

logger = logging.getLogger(__name__)


class TrtInferenceEngine:
    """
    通用 TensorRT 推理引擎
    支持动态 batch、FP16/INT8、异步推理
    """
    
    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 1,
        use_fp16: bool = True,
        use_async: bool = False
    ):
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.use_fp16 = use_fp16
        self.use_async = use_async
        
        self.engine = None
        self.context = None
        self.stream = None
        
        # 输入输出绑定
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_shapes = []
        self.output_shapes = []
        
        if TRT_AVAILABLE and os.path.exists(engine_path):
            self._load_engine()
        else:
            logger.warning(f"Engine not found or TRT unavailable: {engine_path}")
            self._mock_mode = True
    
    def _load_engine(self):
        """加载 TensorRT Engine 文件"""
        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        
        trt_logger = trt.Logger(trt.Logger.INFO)
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 分配输入输出内存
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # 处理动态 shape
            if shape[0] == -1:
                shape = (self.max_batch_size,) + shape[1:]
                self.context.set_input_shape(name, shape)
            
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
            device_mem = cuda.mem_alloc(size)
            
            self.bindings.append(int(device_mem))
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    "name": name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
                self.input_shapes.append(shape)
            else:
                self.outputs.append({
                    "name": name,
                    "host": host_mem,
                    "device": device_mem,
                    "shape": shape,
                    "dtype": dtype
                })
                self.output_shapes.append(shape)
        
        logger.info(f"Engine loaded. Inputs: {len(self.inputs)}, Outputs: {len(self.outputs)}")
    
    def infer(self, input_batch: np.ndarray) -> List[np.ndarray]:
        """
        执行推理
        
        Args:
            input_batch: [B, C, H, W] 归一化后的输入
            
        Returns:
            输出张量列表
        """
        if not TRT_AVAILABLE or getattr(self, '_mock_mode', False):
            return self._mock_infer(input_batch)
        
        batch_size = input_batch.shape[0]
        
        # 复制输入到 GPU
        np.copyto(self.inputs[0]["host"][:batch_size], input_batch.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )
        
        # 设置动态 batch
        if self.input_shapes[0][0] == -1:
            self.context.set_binding_shape(0, (batch_size,) + self.input_shapes[0][1:])
        
        # 执行推理
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 复制输出回 CPU
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            outputs.append(np.copy(out["host"]))
        
        self.stream.synchronize()
        
        # reshape 输出
        results = []
        for i, out in enumerate(self.outputs):
            shape = (batch_size,) + out["shape"][1:]
            results.append(outputs[i][:np.prod(shape)].reshape(shape))
        
        return results
    
    def _mock_infer(self, input_batch: np.ndarray) -> List[np.ndarray]:
        """Mock 推理模式（用于无 GPU 环境调试）"""
        batch_size = input_batch.shape[0]
        # 返回随机输出用于调试
        return [np.random.randn(batch_size, 84, 80, 80).astype(np.float32)]
    
    def warmup(self, iterations: int = 3):
        """预热引擎，避免首次推理延迟"""
        if not self.inputs:
            return
        
        dummy = np.zeros(self.inputs[0]["shape"], dtype=self.inputs[0]["dtype"])
        for _ in range(iterations):
            self.infer(dummy)
        logger.info(f"Warmup completed: {iterations} iterations")
    
    def get_latency_ms(self, input_batch: np.ndarray, runs: int = 10) -> float:
        """测量平均推理延迟"""
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.infer(input_batch)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        return np.mean(times[1:])  # 去掉首次
    
    def __del__(self):
        """释放 CUDA 资源"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.synchronize()
