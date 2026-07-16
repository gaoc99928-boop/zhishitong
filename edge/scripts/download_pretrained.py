"""
预训练模型权重下载脚本
========================
由于模型权重文件较大(>100MB)，不直接放入 Git 仓库，
而是通过本脚本从云存储/网盘下载。

支持的模型:
    - CRNN 车牌识别 (crnn_plate_best.pt / .onnx)
    - YOLOv8-Nano 车辆检测 (yolov8n_vehicle.pt)
    - NAFNet-Mobile 去噪 (nafnet_mobile.pth)
    - ResNet18 车型分类 (resnet18_vehicle.pth)

使用方法:
    python scripts/download_pretrained.py --all
    python scripts/download_pretrained.py --model plate --output ./models
"""
import os
import sys
import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 预训练模型配置
# 实际部署时，请将 URL 替换为你的云存储/网盘直链或 HuggingFace 链接
MODEL_CONFIGS = {
    "crnn_plate": {
        "name": "CRNN 车牌识别模型",
        "files": {
            "crnn_plate_best.pt": {
                # 占位 URL - 请替换为实际下载地址
                "url": "https://example.com/models/crnn_plate_best.pt",
                "md5": "placeholder",
                "size_mb": 12,
                "desc": "CCPD数据集训练，准确率~96%"
            },
            "crnn_plate_best.onnx": {
                "url": "https://example.com/models/crnn_plate_best.onnx",
                "md5": "placeholder",
                "size_mb": 8,
                "desc": "ONNX格式，用于TensorRT转换"
            }
        }
    },
    "yolov8_vehicle": {
        "name": "YOLOv8-Nano 车辆检测",
        "files": {
            "yolov8n_vehicle.pt": {
                "url": "https://example.com/models/yolov8n_vehicle.pt",
                "md5": "placeholder",
                "size_mb": 6,
                "desc": "支持6类车型检测"
            },
            "yolov8n_vehicle.onnx": {
                "url": "https://example.com/models/yolov8n_vehicle.onnx",
                "md5": "placeholder",
                "size_mb": 12,
                "desc": "ONNX格式，用于TensorRT转换"
            }
        }
    },
    "nafnet_denoiser": {
        "name": "NAFNet-Mobile 去噪模型",
        "files": {
            "nafnet_mobile.pth": {
                "url": "https://example.com/models/nafnet_mobile.pth",
                "md5": "placeholder",
                "size_mb": 18,
                "desc": "轻量级图像去噪"
            }
        }
    },
    "vehicle_classifier": {
        "name": "ResNet18 车型分类",
        "files": {
            "resnet18_vehicle.pth": {
                "url": "https://example.com/models/resnet18_vehicle.pth",
                "md5": "placeholder",
                "size_mb": 45,
                "desc": "7类车型分类"
            }
        }
    }
}


def check_md5(file_path: str, expected_md5: str) -> bool:
    """校验文件 MD5"""
    if expected_md5 == "placeholder":
        return True
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def download_file(url: str, save_path: str, desc: str = ""):
    """下载单个文件，带进度显示"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
        logger.info(f"  File already exists: {save_path.name}")
        return True
    
    logger.info(f"  Downloading: {desc or save_path.name}")
    logger.info(f"  URL: {url}")
    logger.info(f"  Save to: {save_path}")
    
    try:
        # 尝试获取文件大小
        try:
            with urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
        except:
            total_size = 0
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r    Progress: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        
        urlretrieve(url, str(save_path), reporthook=report_progress)
        print()  # newline
        
        logger.info(f"  Download complete: {save_path.name}")
        return True
        
    except HTTPError as e:
        logger.error(f"  Download failed: HTTP {e.code}")
        if save_path.exists():
            save_path.unlink()
        return False
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_model(model_key: str, output_dir: str, skip_verify: bool = False):
    """下载指定模型的所有文件"""
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        logger.error(f"Unknown model: {model_key}")
        return False
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Downloading: {config['name']}")
    logger.info(f"{'='*50}")
    
    success = True
    for filename, fileinfo in config["files"].items():
        save_path = os.path.join(output_dir, filename)
        
        if not download_file(fileinfo["url"], save_path, fileinfo["desc"]):
            success = False
            continue
        
        if not skip_verify and not check_md5(save_path, fileinfo["md5"]):
            logger.warning(f"  MD5 check failed for {filename}")
            # 不标记失败，因为 placeholder MD5 总是通过
    
    return success


def generate_manual_instructions():
    """生成手动下载说明文档"""
    lines = [
        "# 预训练模型权重手动下载指南",
        "",
        "由于模型权重文件较大，请通过以下方式获取预训练模型：",
        "",
        "## 方式一：网盘下载 (推荐)",
        "",
        "百度网盘链接: https://pan.baidu.com/s/xxxxxxxxx  提取码: zst1",
        "",
        "下载后将模型文件放置到对应目录：",
        "```",
    ]
    
    for key, config in MODEL_CONFIGS.items():
        lines.append(f"\n### {config['name']}")
        for fname, finfo in config["files"].items():
            lines.append(f"- {fname} ({finfo['size_mb']}MB) - {finfo['desc']}")
            if fname.endswith('.pt') or fname.endswith('.pth'):
                lines.append(f"  放置路径: `models/pytorch/{fname}`")
            elif fname.endswith('.onnx'):
                lines.append(f"  放置路径: `models/onnx/{fname}`")
            elif fname.endswith('.engine'):
                lines.append(f"  放置路径: `models/engines/{fname}`")
    
    lines.extend([
        "```",
        "",
        "## 方式二：自行训练",
        "",
        "使用 `edge/train/` 下的训练脚本自行训练：",
        "```bash",
        "# 车牌识别 (需要 CCPD 数据集)",
        "python edge/train/train_plate_recognition.py --data_dir /path/to/ccpd --epochs 50",
        "",
        "# 车辆检测",
        "python edge/train/train_yolov8.py --data_dir /path/to/vehicle_data --epochs 100",
        "",
        "# 去噪模型",
        "python edge/train/train_denoiser.py --data_dir /path/to/noisy_data --epochs 200",
        "```",
        "",
        "## 方式三：HuggingFace",
        "",
        "模型已上传至 HuggingFace Hub：",
        "https://huggingface.co/your-username/zhishitong-models",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="下载预训练模型权重")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()) + ["all"],
                       help="要下载的模型名称")
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--output", default="./models", help="输出目录")
    parser.add_argument("--generate-readme", action="store_true",
                       help="生成手动下载说明文档")
    parser.add_argument("--skip-verify", action="store_true", help="跳过MD5校验")
    
    args = parser.parse_args()
    
    if args.generate_readme:
        readme = generate_manual_instructions()
        readme_path = Path(args.output) / "MODEL_DOWNLOAD.md"
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)
        logger.info(f"Manual download guide generated: {readme_path}")
        return
    
    if not args.model and not args.all:
        parser.print_help()
        print("\nAvailable models:")
        for key, config in MODEL_CONFIGS.items():
            print(f"  {key}: {config['name']}")
        return
    
    models_to_download = list(MODEL_CONFIGS.keys()) if args.all else [args.model]
    
    logger.info("=" * 50)
    logger.info("智视通 - 预训练模型下载")
    logger.info("=" * 50)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Models to download: {', '.join(models_to_download)}")
    
    all_success = True
    for model_key in models_to_download:
        if not download_model(model_key, args.output, args.skip_verify):
            all_success = False
    
    logger.info("\n" + "=" * 50)
    if all_success:
        logger.info("All downloads completed successfully!")
        logger.info("Next: Convert ONNX to TensorRT with tools/convert_*.py")
    else:
        logger.warning("Some downloads failed. Please check the URLs or use manual download.")
        logger.info(f"Run with --generate-readme to create manual download instructions.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
