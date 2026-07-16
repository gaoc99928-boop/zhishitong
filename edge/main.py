"""
智视通 — 边缘计算节点主程序
=====================================
入口文件，整合视频流、AI 流水线、数据存储、MQTT 上报

运行方式:
    python main.py --config config.yaml
    python main.py --source test_video.mp4 --display
    python main.py --mode benchmark

硬件要求:
    - NVIDIA Jetson Orin Nano
    - JetPack 6.0, CUDA 12.2, TensorRT 8.6
    - 可选: DeepStream 6.3 (用于硬件解码)
"""
import os
import sys
import argparse
import logging
import signal
import time
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.processor import VehiclePipeline
from edge_node.data_store import EdgeDataStore
from edge_node.mqtt_client import EdgeMqttClient
from utils.image_utils import create_comparison_view


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


class EdgeNode:
    """
    边缘计算节点控制器
    
    职责:
    - 管理视频流输入 (RTSP/USB/文件)
    - 调度 AI 流水线处理每一帧
    - 持久化结果到 SQLite
    - 通过 MQTT 上报云端
    - 处理优雅退出
    """
    
    def __init__(self, config_path: str = "./config.yaml"):
        self.config_path = config_path
        self.running = False
        
        # 初始化各组件
        self.pipeline = VehiclePipeline(config_path)
        self.config = self.pipeline.config
        
        edge_cfg = self.config.get("edge", {})
        
        # 数据存储
        self.data_store = EdgeDataStore(
            db_path=edge_cfg.get("sqlite", {}).get("db_path", "./data/records.db"),
            retention_days=edge_cfg.get("sqlite", {}).get("retention_days", 7)
        )
        
        # MQTT 客户端
        mqtt_cfg = edge_cfg.get("mqtt", {})
        self.mqtt = None
        if mqtt_cfg.get("enable", False):
            self.mqtt = EdgeMqttClient(
                broker_host=mqtt_cfg.get("broker_host", "localhost"),
                broker_port=mqtt_cfg.get("broker_port", 1883),
                topic_prefix=mqtt_cfg.get("topic_prefix", "zhishitong/node"),
                client_id=mqtt_cfg.get("client_id", "edge-01"),
                qos=mqtt_cfg.get("qos", 1)
            )
        
        # 统计信息
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _create_video_source(self, source: str):
        """
        创建视频源
        
        支持:
        - RTSP 流: rtsp://...
        - USB 摄像头: /dev/video0 或 0
        - 视频文件: *.mp4, *.avi
        """
        if source.startswith("rtsp://"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        elif source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # 设置缓冲区大小 (减少延迟)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    
    def run(self, source: str = "0", display: bool = False, save_path: str = None):
        """
        运行主循环
        
        Args:
            source: 视频源
            display: 是否显示预览窗口
            save_path: 保存处理后的视频路径
        """
        logging.info(f"Starting edge node: {self.config.get('system', {}).get('name')}")
        logging.info(f"Video source: {source}")
        
        # 连接 MQTT
        if self.mqtt:
            if self.mqtt.connect():
                logging.info("MQTT connected")
            else:
                logging.warning("MQTT connection failed, running in offline mode")
        
        # 创建视频源
        cap = self._create_video_source(source)
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logging.info(f"Video properties: {width}x{height} @ {fps}fps")
        
        # 视频写入器 (用于保存)
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        self.running = True
        frame_id = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logging.info("End of video stream")
                    break
                
                frame_id += 1
                self.frame_count += 1
                
                # 处理帧
                result = self.pipeline.process_frame(frame, frame_id)
                
                # 持久化
                if result.get("vehicles"):
                    self.data_store.insert(result)
                
                # MQTT 上报
                if self.mqtt:
                    self.mqtt.publish_detection(result)
                
                # 显示
                if display:
                    vis = self.pipeline.draw_results(frame, result)
                    cv2.imshow("智视通 - 边缘识别", vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s"):
                        # 手动保存当前帧
                        save_file = f"./snapshots/frame_{frame_id}.jpg"
                        Path("./snapshots").mkdir(exist_ok=True)
                        cv2.imwrite(save_file, vis)
                        logging.info(f"Snapshot saved: {save_file}")
                
                # 保存视频
                if writer:
                    writer.write(frame)
                
                # 打印 FPS
                if frame_id % 30 == 0:
                    elapsed = time.time() - self.last_fps_time
                    current_fps = 30 / elapsed
                    self.last_fps_time = time.time()
                    logging.info(f"Frame {frame_id}, FPS: {current_fps:.1f}, "
                               f"Latency: {result.get('processing_time_ms', 0):.1f}ms, "
                               f"Vehicles: {len(result.get('vehicles', []))}")
        
        finally:
            # 清理资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            if self.mqtt:
                self.mqtt.disconnect()
            
            # 打印统计
            stats = self.pipeline.get_stats()
            logging.info("=== Processing Statistics ===")
            logging.info(f"Total frames: {stats['total_frames']}")
            logging.info(f"Total vehicles: {stats['total_vehicles']}")
            logging.info(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
            logging.info(f"Avg vehicles/frame: {stats['avg_vehicles_per_frame']:.2f}")
            logging.info("Edge node stopped")
    
    def run_benchmark(self, image_path: str = None, runs: int = 100):
        """
        运行性能基准测试
        
        Args:
            image_path: 测试图像路径，None 则生成模拟图像
            runs: 测试轮数
        """
        logging.info(f"Running benchmark ({runs} iterations)...")
        
        if image_path and Path(image_path).exists():
            test_image = cv2.imread(image_path)
        else:
            # 生成模拟图像
            test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # 预热
        logging.info("Warming up...")
        for _ in range(5):
            self.pipeline.process_frame(test_image)
        
        # 基准测试
        latencies = []
        for i in range(runs):
            t0 = time.perf_counter()
            result = self.pipeline.process_frame(test_image, i)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        
        # 统计
        latencies = np.array(latencies)
        logging.info("=== Benchmark Results ===")
        logging.info(f"Runs: {runs}")
        logging.info(f"Mean latency: {np.mean(latencies):.2f}ms")
        logging.info(f"Std deviation: {np.std(latencies):.2f}ms")
        logging.info(f"Min latency: {np.min(latencies):.2f}ms")
        logging.info(f"Max latency: {np.max(latencies):.2f}ms")
        logging.info(f"P50 latency: {np.percentile(latencies, 50):.2f}ms")
        logging.info(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
        logging.info(f"P99 latency: {np.percentile(latencies, 99):.2f}ms")
        logging.info(f"Estimated FPS: {1000 / np.mean(latencies):.1f}")


def main():
    parser = argparse.ArgumentParser(description="智视通 - 边缘计算节点")
    parser.add_argument("--config", default="./config.yaml", help="配置文件路径")
    parser.add_argument("--source", default="0", help="视频源 (RTSP/文件/摄像头ID)")
    parser.add_argument("--display", action="store_true", help="显示预览窗口")
    parser.add_argument("--save", default=None, help="保存处理后的视频路径")
    parser.add_argument("--mode", default="run", choices=["run", "benchmark"], help="运行模式")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    
    node = EdgeNode(args.config)
    
    if args.mode == "benchmark":
        node.run_benchmark(runs=100)
    else:
        node.run(source=args.source, display=args.display, save_path=args.save)


if __name__ == "__main__":
    main()
