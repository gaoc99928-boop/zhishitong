# 智视通 — 边缘计算节点

> 交通路口车辆图像增强与识别系统，基于 NVIDIA Jetson Orin Nano 的边缘 AI 解决方案。

## 系统架构

```
视频流输入 (RTSP/USB/文件)
    ↓
[图像质量评估]
    ↓
[去噪增强] (NAFNet-Mobile / BM3D 回退)
    ↓
[车辆检测] (YOLOv8-Nano + TensorRT)
    ↓
[车牌识别] (LPRNet + TensorRT)
    ↓
[结构化输出] → SQLite 本地存储 + MQTT 云端上报
```

## 硬件要求

| 组件 | 规格 |
|------|------|
| 主控 | NVIDIA Jetson Orin Nano |
| 算力 | 40 TOPS INT8 |
| 功耗 | 15W |
| 内存 | 8GB LPDDR5 |
| 系统 | JetPack 6.0 (L4T 36.2) |
| CUDA | 12.2 |
| TensorRT | 8.6 |

## 快速开始

### 1. 环境准备

```bash
# Jetson 上安装依赖
sudo apt update
sudo apt install -y python3-pip python3-opencv libopencv-dev

# 安装 Python 依赖
pip3 install -r requirements.txt
```

### 2. 模型准备

```bash
# 下载预训练 ONNX 模型并转换为 TensorRT Engine
python3 tools/convert_yolov8_trt.py --weights yolov8n_vehicle.onnx --fp16
python3 tools/convert_lprnet_trt.py --weights lprnet_chinese.onnx --fp16
```

### 3. 运行

```bash
# 实时识别 (摄像头)
python3 main.py --source 0 --display

# 处理视频文件
python3 main.py --source ./test.mp4 --save ./output.mp4

# RTSP 流
python3 main.py --source rtsp://192.168.1.10/stream1

# 性能基准测试
python3 main.py --mode benchmark --runs 100
```

## 项目结构

```
edge/
├── main.py              # 主程序入口
├── config.yaml          # 配置文件
├── requirements.txt     # Python 依赖
├── models/
│   ├── trt_engine.py    # TensorRT 推理引擎封装
│   └── engines/         # .engine 模型文件
├── pipeline/
│   ├── denoiser.py      # 图像去噪 (NAFNet/BM3D)
│   ├── detector.py      # 车辆检测 (YOLOv8)
│   ├── recognizer.py    # 车牌识别 (LPRNet)
│   └── processor.py     # 流水线 orchestrator
├── edge_node/
│   ├── mqtt_client.py   # MQTT 通信
│   └── data_store.py    # SQLite 本地存储
├── utils/
│   └── image_utils.py   # 图像处理工具
└── tools/
    ├── convert_yolov8_trt.py
    ├── convert_lprnet_trt.py
    └── benchmark.py
```

## 配置文件说明

`config.yaml` 关键参数:

```yaml
models:
  detector:
    engine_path: "./models/engines/yolov8n_vehicle.engine"
    conf_threshold: 0.45
  
  recognizer:
    engine_path: "./models/engines/lprnet_chinese.engine"
    
  denoiser:
    mode: "on_demand"  # real_time / on_demand / off

edge:
  mqtt:
    enable: true
    broker_host: "192.168.1.200"
    
  sqlite:
    retention_days: 7
```

## 性能指标

| 场景 | 延迟 | 备注 |
|------|------|------|
| 检测单帧 | ~50ms | YOLOv8-Nano FP16 |
| 识别车牌 | ~30ms | LPRNet FP16 |
| 去噪增强 | ~50ms | NAFNet-Mobile |
| 完整流水线 | ~180ms | 单路 |
| 4路并行 | 15-20 fps | 总吞吐量 |

## 技术栈

- **推理加速**: TensorRT 8.6 (FP16/INT8)
- **视频解码**: NVDEC (硬件加速)
- **检测模型**: YOLOv8-Nano
- **识别模型**: LPRNet
- **去噪模型**: NAFNet-Mobile
- **数据存储**: SQLite + 7天滚动
- **通信协议**: MQTT (QoS 1)
