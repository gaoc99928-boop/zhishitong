"""
LPRNet ONNX → TensorRT Engine 转换脚本
适配中文车牌识别模型

Usage:
    python convert_lprnet_trt.py --weights lprnet.onnx --fp16
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
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


def build_engine(onnx_path: str, engine_path: str, fp16: bool = False) -> bool:
    """构建 LPRNet TensorRT Engine"""
    if not TRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False
    
    logger.info(f"Building LPRNet engine from: {onnx_path}")
    
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            return False
    
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * (1 << 20)
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    engine = builder.build_engine(network, config)
    if engine is None:
        return False
    
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    
    logger.info(f"Engine saved: {engine_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="ONNX 路径")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fp16", action="store_true")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.weights.replace(".onnx", ".engine")
    
    # 打印 trtexec 命令
    cmd = f"trtexec --onnx={args.weights} --saveEngine={args.output}"
    if args.fp16:
        cmd += " --fp16"
    print(f"\n等效命令: {cmd}\n")
    
    if build_engine(args.weights, args.output, args.fp16):
        logger.info("Success!")
    else:
        logger.info("Please run the trtexec command above on Jetson")


if __name__ == "__main__":
    main()
