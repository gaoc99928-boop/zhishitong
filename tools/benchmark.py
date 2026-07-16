"""
智视通 — 性能基准测试工具
测量各模块推理延迟和端到端吞吐量
"""
import os
import sys
import time
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "edge"))

from pipeline.denoiser import ImageDenoiser
from pipeline.detector import VehicleDetector
from pipeline.recognizer import LicensePlateRecognizer
from models.trt_engine import TrtInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def benchmark_module(name: str, infer_func, input_data, warmup: int = 5, runs: int = 100):
    """基准测试单个模块"""
    # 预热
    for _ in range(warmup):
        infer_func(input_data)
    
    # 测试
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        infer_func(input_data)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    times = np.array(times)
    return {
        "module": name,
        "runs": runs,
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "min_ms": round(float(np.min(times)), 2),
        "max_ms": round(float(np.max(times)), 2),
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "p99_ms": round(float(np.percentile(times, 99)), 2),
        "fps": round(1000 / float(np.mean(times)), 1)
    }


def run_full_benchmark(config: dict):
    """运行完整基准测试"""
    logger.info("=" * 60)
    logger.info("智视通 — 性能基准测试")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 生成测试图像
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    test_plate = np.random.randint(0, 255, (24, 94, 3), dtype=np.uint8)
    
    results = []
    
    # 1. 去噪模块
    logger.info("\n[1/4] 测试去噪模块...")
    denoiser = ImageDenoiser(mode="on_demand")
    denoiser_input = cv2.resize(test_image, (256, 256))
    denoiser_input = cv2.cvtColor(denoiser_input, cv2.COLOR_BGR2RGB)
    denoiser_input = denoiser_input.astype(np.float32) / 255.0
    denoiser_input = np.transpose(denoiser_input, (2, 0, 1))[np.newaxis, ...]
    
    r = benchmark_module(
        "Denoiser (BM3D fallback)",
        lambda x: denoiser._denoise_bm3d(test_image),
        test_image,
        runs=20
    )
    results.append(r)
    logger.info(f"  Mean: {r['mean_ms']}ms, FPS: {r['fps']}")
    
    # 2. 检测模块
    logger.info("\n[2/4] 测试检测模块...")
    # Mock 模式
    detector = VehicleDetector(
        engine_path="mock",
        input_size=(640, 640),
        labels=["car", "suv", "truck", "bus", "van", "taxi"]
    )
    detector._mock_mode = True
    
    r = benchmark_module(
        "Detector (YOLOv8-Nano)",
        lambda x: detector.detect(test_image),
        test_image,
        runs=50
    )
    results.append(r)
    logger.info(f"  Mean: {r['mean_ms']}ms, FPS: {r['fps']}")
    
    # 3. 识别模块
    logger.info("\n[3/4] 测试识别模块...")
    recognizer = LicensePlateRecognizer(
        engine_path="mock",
        input_size=(94, 24),
        charset="0123456789ABCDEFGHJKLMNPQRSTUVWXYZ沪京粤苏浙川鲁"
    )
    recognizer._mock_mode = True
    
    r = benchmark_module(
        "Recognizer (LPRNet)",
        lambda x: recognizer.recognize(test_plate),
        test_plate,
        runs=100
    )
    results.append(r)
    logger.info(f"  Mean: {r['mean_ms']}ms, FPS: {r['fps']}")
    
    # 4. 完整流水线
    logger.info("\n[4/4] 测试完整流水线...")
    from pipeline.processor import VehiclePipeline
    pipeline = VehiclePipeline()
    
    r = benchmark_module(
        "Full Pipeline",
        lambda x: pipeline.process_frame(test_image),
        test_image,
        runs=30
    )
    results.append(r)
    logger.info(f"  Mean: {r['mean_ms']}ms, FPS: {r['fps']}")
    
    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    for r in results:
        logger.info(f"\n{r['module']}:")
        logger.info(f"  平均延迟: {r['mean_ms']}ms ± {r['std_ms']}ms")
        logger.info(f"  P50/P95/P99: {r['p50_ms']}ms / {r['p95_ms']}ms / {r['p99_ms']}ms")
        logger.info(f"  等效 FPS: {r['fps']}")
    
    # 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": "mock_benchmark",
        "results": results
    }
    
    output_path = "benchmark_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n报告已保存: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100, help="每模块测试轮数")
    parser.add_argument("--output", default="benchmark_report.json", help="报告输出路径")
    
    args = parser.parse_args()
    
    run_full_benchmark({})


if __name__ == "__main__":
    main()
