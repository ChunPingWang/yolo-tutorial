"""
YOLOv11 圖片物件偵測腳本
支援本地圖片或 URL 輸入，輸出標注圖片與 JSON 結果
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 圖片偵測")
    parser.add_argument(
        "--source",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="圖片路徑或 URL",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo11n.pt",
        help="模型權重路徑",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="信心閾值",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="儲存 JSON 格式結果",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="輸出目錄",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 載入模型
    print(f"載入模型: {args.weights}")
    model = YOLO(args.weights)

    # 執行推論
    print(f"偵測圖片: {args.source}")
    start = time.time()
    results = model(args.source, conf=args.conf)
    elapsed = time.time() - start
    print(f"推論完成，耗時: {elapsed:.3f} 秒")

    # 處理結果
    for result in results:
        # 儲存標注圖片
        annotated = result.plot()
        import cv2

        out_path = output_dir / "result.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"標注圖片已儲存: {out_path}")

        # 儲存 JSON
        if args.save_json:
            detections = []
            for box in result.boxes:
                detections.append(
                    {
                        "class": result.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox_xyxy": [round(x, 1) for x in box.xyxy[0].tolist()],
                    }
                )

            json_result = {
                "file": args.source,
                "image_size": list(result.orig_shape),
                "inference_time_ms": round(elapsed * 1000, 1),
                "detections": detections,
            }

            json_path = output_dir / "result.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False, indent=2)
            print(f"JSON 結果已儲存: {json_path}")
            print(f"偵測到 {len(detections)} 個物體")


if __name__ == "__main__":
    main()
