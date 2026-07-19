"""
YOLOv11 TensorRT 模型轉換腳本
將 PyTorch 模型匯出為 TensorRT engine 以加速推論
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 TensorRT 匯出")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo11n.pt",
        help="PyTorch 模型權重路徑",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="使用 FP16 半精度",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="使用 INT8 量化（需要校準資料）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="INT8 校準資料集 YAML (例: coco128.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="輸入影像尺寸",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="TensorRT workspace 大小 (GB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[資訊] 模型不存在，將自動下載: {args.weights}")

    # 載入模型
    print(f"載入模型: {args.weights}")
    model = YOLO(args.weights)

    # 設定匯出參數
    export_args = {
        "format": "engine",
        "imgsz": args.imgsz,
        "half": args.half or args.int8,
        "int8": args.int8,
        "workspace": args.workspace,
    }

    if args.int8:
        if args.data is None:
            print("[錯誤] INT8 量化需要指定 --data 校準資料集")
            return
        export_args["data"] = args.data
        print(f"INT8 校準資料集: {args.data}")

    precision = "INT8" if args.int8 else ("FP16" if args.half else "FP32")
    print(f"匯出精度: {precision}")
    print(f"輸入尺寸: {args.imgsz}x{args.imgsz}")
    print(f"Workspace: {args.workspace} GB")
    print()
    print("開始匯出 TensorRT engine（可能需要 2-5 分鐘）...")

    start = time.time()
    engine_path = model.export(**export_args)
    elapsed = time.time() - start

    print(f"\n匯出完成！")
    print(f"Engine 路徑: {engine_path}")
    print(f"匯出耗時: {elapsed:.1f} 秒")

    # 驗證 engine
    print("\n驗證 engine 推論...")
    engine_model = YOLO(engine_path)
    results = engine_model("https://ultralytics.com/images/bus.jpg", verbose=False)
    num_detections = len(results[0].boxes)
    print(f"驗證成功，偵測到 {num_detections} 個物體")


if __name__ == "__main__":
    main()
