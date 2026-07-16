"""
通用 ONNX 导出工具
==================
支持从 PyTorch 模型导出 ONNX，为 TensorRT 转换做准备
"""
import argparse
import logging
from pathlib import Path

try:
    import torch
    import torch.onnx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch torchvision")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_yolov8_to_onnx(weights_path: str, output_path: str, imgsz: int = 640):
    """
    导出 YOLOv8 检测模型到 ONNX
    
    Args:
        weights_path: YOLOv8 .pt 权重文件路径
        output_path: 输出 .onnx 文件路径
        imgsz: 输入图像尺寸
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    
    logger.info(f"Loading YOLOv8 model: {weights_path}")
    model = YOLO(weights_path)
    
    logger.info(f"Exporting to ONNX (imgsz={imgsz})...")
    model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=True,
        simplify=True,
        opset=17
    )
    
    # ultralytics 会自动生成同名 onnx 文件
    auto_onnx = Path(weights_path).with_suffix(".onnx")
    if auto_onnx.exists():
        auto_onnx.rename(output_path)
        logger.info(f"ONNX saved: {output_path}")


def export_lprnet_to_onnx(model, output_path: str, input_size: tuple = (1, 3, 24, 94)):
    """
    导出 LPRNet 模型到 ONNX
    
    Args:
        model: PyTorch LPRNet 模型实例
        output_path: 输出 .onnx 文件路径
        input_size: (batch, channels, height, width)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")
    
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    logger.info(f"LPRNet ONNX saved: {output_path}")


def export_classifier_to_onnx(model, output_path: str, input_size: tuple = (1, 3, 224, 224)):
    """
    导出分类模型到 ONNX (ResNet18/ShuffleNetV2)
    
    Args:
        model: PyTorch 分类模型实例
        output_path: 输出 .onnx 文件路径
        input_size: (batch, channels, height, width)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")
    
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    logger.info(f"Classifier ONNX saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="导出 PyTorch 模型到 ONNX")
    parser.add_argument("--model", required=True, choices=["yolov8", "lprnet", "classifier"])
    parser.add_argument("--weights", required=True, help="PyTorch 权重文件路径")
    parser.add_argument("--output", required=True, help="输出 ONNX 文件路径")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLOv8 输入尺寸")
    
    args = parser.parse_args()
    
    if args.model == "yolov8":
        export_yolov8_to_onnx(args.weights, args.output, args.imgsz)
    else:
        logger.error(f"Model type '{args.model}' requires manual model loading. "
                     f"Use the specific export function in your training script.")


if __name__ == "__main__":
    main()
