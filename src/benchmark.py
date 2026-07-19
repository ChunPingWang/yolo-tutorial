"""
YOLOv11 效能基準測試腳本
測量推論延遲、吞吐量與百分位數統計
"""

import argparse
import time

import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 效能基準測試")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo11n.pt",
        help="模型權重路徑 (.pt 或 .engine)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="輸入影像尺寸",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="暖機次數（排除 JIT/cuDNN 初始化開銷）",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="正式測試迭代次數",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="裝置 (0=GPU, cpu=CPU)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print(" YOLOv11 效能基準測試")
    print("=" * 50)

    # 顯示系統資訊
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"模型: {args.weights}")
    print(f"輸入尺寸: {args.imgsz}x{args.imgsz}")
    print(f"暖機: {args.warmup} 次 | 測試: {args.iters} 次")
    print()

    # 載入模型
    model = YOLO(args.weights)

    # 建立隨機測試影像
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    # 暖機階段
    print("暖機中...")
    for _ in range(args.warmup):
        model(dummy, verbose=False)

    # 正式測試
    print("測試中...")
    latencies = []

    for i in range(args.iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        model(dummy, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    # 統計結果
    latencies = np.array(latencies)
    avg = np.mean(latencies)
    std = np.std(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    fps = 1000.0 / avg

    print()
    print("=" * 50)
    print(" 測試結果")
    print("=" * 50)
    print(f"  平均延遲:  {avg:.2f} ms (±{std:.2f})")
    print(f"  P50 延遲:  {p50:.2f} ms")
    print(f"  P95 延遲:  {p95:.2f} ms")
    print(f"  P99 延遲:  {p99:.2f} ms")
    print(f"  吞吐量:    {fps:.1f} FPS")
    print(f"  最快:      {np.min(latencies):.2f} ms")
    print(f"  最慢:      {np.max(latencies):.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
