"""
车辆检测模型训练脚本 (YOLOv8-Nano)
=====================================
基于 Ultralytics YOLOv8，针对路口车辆检测场景微调

使用方法:
    # 从预训练权重微调
    python train_yolov8.py --data vehicle_data.yaml --epochs 100 --imgsz 640
    
    # 从头训练 (不推荐，除非数据量>10k)
    python train_yolov8.py --data vehicle_data.yaml --epochs 200 --pretrained false

数据集格式 (YOLOv8):
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml

数据标注格式:
    class_id x_center y_center width height (归一化 0-1)
    0 0.5123 0.6789 0.2341 0.1876
"""
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolov8(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    model_size: str = "n",  # n, s, m, l, x
    pretrained: bool = True,
    device: str = "0",
    project: str = "./runs/detect",
    name: str = "vehicle_detection",
    lr0: float = 0.01,
    lrf: float = 0.01,
    augment: bool = True,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    patience: int = 20
):
    """
    训练 YOLOv8 车辆检测模型
    
    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批量大小
        model_size: 模型尺寸 (n=Nano, s=Small, m=Medium)
        pretrained: 是否使用预训练权重
        device: GPU 设备ID
        project: 输出项目目录
        name: 实验名称
        lr0: 初始学习率
        lrf: 最终学习率因子
        augment: 是否启用数据增强
        mosaic: Mosaic 增强概率
        mixup: MixUp 增强概率
        patience: 早停耐心值
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # 选择预训练模型
    if pretrained:
        model_path = f"yolov8{model_size}.pt"
        logger.info(f"Loading pretrained model: {model_path}")
    else:
        model_path = f"yolov8{model_size}.yaml"
        logger.info(f"Creating model from scratch: {model_path}")
    
    model = YOLO(model_path)
    
    # 类别名称 (6类车辆)
    class_names = ["car", "suv", "truck", "bus", "van", "taxi"]
    logger.info(f"Classes: {class_names}")
    
    # 训练参数
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "lr0": lr0,
        "lrf": lrf,
        "patience": patience,
        "save": True,
        "save_period": 10,
        "exist_ok": True,
        "pretrained": pretrained,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "momentum": 0.937,
        "cos_lr": True,
        "close_mosaic": epochs - 10,  # 最后10轮关闭mosaic
        "amp": True,  # 混合精度训练
    }
    
    if augment:
        train_args.update({
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": mosaic,
            "mixup": mixup,
            "copy_paste": 0.1,
        })
    
    logger.info("=" * 50)
    logger.info("Starting YOLOv8 training")
    logger.info("=" * 50)
    
    # 开始训练
    results = model.train(**train_args)
    
    # 验证最佳模型
    logger.info("\nValidating best model...")
    metrics = model.val()
    
    logger.info(f"\nBest mAP50: {metrics.box.map50:.4f}")
    logger.info(f"Best mAP50-95: {metrics.box.map:.4f}")
    
    # 导出 ONNX
    logger.info("\nExporting to ONNX...")
    model.export(format="onnx", dynamic=True, simplify=True)
    
    # 导出 TensorRT (如果在 Jetson 上)
    if os.path.exists("/usr/src/tensorrt"):
        logger.info("Exporting to TensorRT Engine...")
        model.export(format="engine", half=True, workspace=2)
    
    logger.info("\n" + "=" * 50)
    logger.info("Training complete!")
    logger.info(f"Best model: {project}/{name}/weights/best.pt")
    logger.info(f"ONNX model: {project}/{name}/weights/best.onnx")
    logger.info("=" * 50)
    
    return results


def create_sample_data_yaml(output_path: str):
    """生成示例数据集配置文件"""
    yaml_content = """# 智视通 - 车辆检测数据集配置
train: ./dataset/train/images
val: ./dataset/val/images
test: ./dataset/test/images

# 类别数量
nc: 6

# 类别名称
names:
  0: car      # 轿车
  1: suv      # SUV
  2: truck    # 大型货车
  3: bus      # 公交车
  4: van      # 面包车
  5: taxi     # 出租车

# 数据增强配置 (可选)
roboflow:
  workspace: zhishitong
  project: vehicle-detection
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/zhishitong/vehicle-detection/dataset/1
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info(f"Sample data YAML created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv8 车辆检测模型")
    parser.add_argument("--data", required=True, help="数据集 YAML 配置文件")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="./runs/detect")
    parser.add_argument("--name", default="vehicle_detection")
    parser.add_argument("--create-sample", action="store_true", help="生成示例 YAML")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data_yaml("vehicle_data.yaml")
        return
    
    train_yolov8(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model_size,
        pretrained=not args.no_pretrained,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
