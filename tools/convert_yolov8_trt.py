"""
YOLOv8-Nano ONNX → TensorRT Engine 转换脚本
适配 Jetson Orin Nano, 支持 FP16/INT8

Usage:
    python convert_yolov8_trt.py --weights yolov8n.onnx --fp16
    python convert_yolov8_trt.py --weights yolov8n.onnx --int8 --calib-data ./calib/
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not available, script will generate command only")


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_mb: int = 512
) -> bool:
    """
    构建 TensorRT Engine
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 Engine 路径
        fp16: 启用 FP16 精度
        int8: 启用 INT8 精度 (需要校准数据)
        max_batch_size: 最大 batch 大小
        workspace_mb: 工作区大小 (MB)
    """
    if not TRT_AVAILABLE:
        logger.error("TensorRT Python API not available")
        return False
    
    logger.info(f"Building engine from: {onnx_path}")
    logger.info(f"FP16: {fp16}, INT8: {int8}, Max batch: {max_batch_size}")
    
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    
    # 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(error)}")
            return False
    
    # 配置 builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")
    
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: 配置 INT8 校准器
        logger.warning("INT8 calibration not implemented in this script")
    
    # 动态 batch 配置
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    
    min_shape = (1,) + input_shape[1:]
    opt_shape = (max_batch_size // 2,) + input_shape[1:]
    max_shape = (max_batch_size,) + input_shape[1:]
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 构建 engine
    logger.info("Building engine... (this may take a few minutes)")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        logger.error("Engine build failed")
        return False
    
    # 保存 engine
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    logger.info(f"Engine saved: {engine_path}")
    logger.info(f"Engine size: {Path(engine_path).stat().st_size / (1<<20):.2f} MB")
    
    return True


def print_trtexec_command(onnx_path: str, fp16: bool = False, int8: bool = False):
    """打印 trtexec 命令行等价命令"""
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={onnx_path.replace('.onnx', '.engine')}"
    
    if fp16:
        cmd += " --fp16"
    if int8:
        cmd += " --int8"
    
    cmd += " --workspace=512 --minShapes=input:1x3x640x640"
    cmd += " --optShapes=input:4x3x640x640 --maxShapes=input:8x3x640x640"
    
    print("\n=== trtexec 等效命令 ===")
    print(cmd)
    print("========================\n")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX to TensorRT Converter")
    parser.add_argument("--weights", required=True, help="ONNX 权重文件路径")
    parser.add_argument("--output", default=None, help="输出 Engine 路径")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16")
    parser.add_argument("--int8", action="store_true", help="启用 INT8")
    parser.add_argument("--max-batch", type=int, default=4, help="最大 batch 大小")
    parser.add_argument("--workspace", type=int, default=512, help="工作区大小 (MB)")
    parser.add_argument("--trtexec-only", action="store_true", help="仅打印 trtexec 命令")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.weights.replace(".onnx", ".engine")
    
    # 打印 trtexec 命令
    print_trtexec_command(args.weights, args.fp16, args.int8)
    
    if args.trtexec_only:
        logger.info("Use the command above on Jetson to convert")
        return
    
    # 尝试用 Python API 转换
    if build_engine(
        onnx_path=args.weights,
        engine_path=args.output,
        fp16=args.fp16,
        int8=args.int8,
        max_batch_size=args.max_batch,
        workspace_mb=args.workspace
    ):
        logger.info("Conversion successful!")
    else:
        logger.error("Conversion failed. Use trtexec command above on Jetson.")
        sys.exit(1)


if __name__ == "__main__":
    main()
