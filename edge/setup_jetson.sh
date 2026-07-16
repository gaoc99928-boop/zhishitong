#!/bin/bash
# =============================================================================
# 智视通 — Jetson Orin Nano 环境安装脚本
# =============================================================================
# 运行方式: bash setup_jetson.sh
# 前置条件: JetPack 6.0 已刷写, CUDA/cuDNN/TensorRT 已预装

set -e

echo "============================================"
echo "  智视通边缘节点环境安装"
echo "  目标设备: NVIDIA Jetson Orin Nano"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. 系统依赖
# ---------------------------------------------------------------------------
echo "[1/6] 安装系统依赖..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    gstreamer1.0-tools libnvinfer8 python3-libnvinfer \
    libnvvpi3 vpi3-dev libopencv-dev \
    mosquitto mosquitto-clients

# ---------------------------------------------------------------------------
# 2. Python 虚拟环境
# ---------------------------------------------------------------------------
echo "[2/6] 创建 Python 虚拟环境..."
python3 -m venv venv --system-site-packages
source venv/bin/activate

# ---------------------------------------------------------------------------
# 3. Python 依赖
# ---------------------------------------------------------------------------
echo "[3/6] 安装 Python 包..."
pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 4. 创建目录结构
# ---------------------------------------------------------------------------
echo "[4/6] 创建目录结构..."
mkdir -p data snapshots models/engines models/onnx configs logs

# ---------------------------------------------------------------------------
# 5. 模型引擎准备提示
# ---------------------------------------------------------------------------
echo "[5/6] 模型引擎检查..."
if [ ! -f "models/engines/yolov8n_vehicle.engine" ]; then
    echo "  ⚠  车辆检测引擎未找到，请运行:"
    echo "     python tools/convert_yolov8_trt.py --onnx models/onnx/yolov8n_vehicle.onnx"
fi
if [ ! -f "models/engines/lprnet_chinese.engine" ]; then
    echo "  ⚠  车牌识别引擎未找到，请运行:"
    echo "     python tools/convert_lprnet_trt.py --onnx models/onnx/lprnet_chinese.onnx"
fi

# ---------------------------------------------------------------------------
# 6. 系统服务配置 (可选)
# ---------------------------------------------------------------------------
echo "[6/6] 配置系统服务..."
cat << 'EOF' | sudo tee /etc/systemd/system/zhishitong-edge.service
[Unit]
Description=智视通边缘计算节点
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/zhishitong/edge
Environment="PYTHONPATH=/home/$USER/zhishitong/edge"
ExecStart=/home/$USER/zhishitong/edge/venv/bin/python main.py --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "============================================"
echo "  安装完成!"
echo "============================================"
echo "  启动命令:"
echo "    source venv/bin/activate"
echo "    python main.py --source 0 --display"
echo ""
echo "  注册为系统服务:"
echo "    sudo systemctl enable zhishitong-edge"
echo "    sudo systemctl start zhishitong-edge"
echo ""
echo "  查看日志:"
echo "    sudo journalctl -u zhishitong-edge -f"
echo "============================================"
