"""
YOLOv11 即時攝影機物件偵測腳本
支援 USB Webcam 與筆電內建鏡頭 (Windows DirectShow)
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 攝影機即時偵測")
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="攝影機 ID (預設: 0)",
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
        "--resolution",
        type=str,
        default="640x480",
        help="攝影機解析度 (寬x高)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    width, height = map(int, args.resolution.split("x"))

    # 載入模型
    print(f"載入模型: {args.weights}")
    model = YOLO(args.weights)

    # 開啟攝影機 (Windows DirectShow 後端)
    print(f"開啟攝影機 ID: {args.source}")
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("[錯誤] 無法開啟攝影機，請確認裝置是否連接")
        return

    print("即時偵測中... 按 'q' 退出")
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 無法讀取畫面")
            break

        # 執行推論
        start = time.time()
        results = model(frame, conf=args.conf, verbose=False)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)

        # 繪製結果
        annotated = results[0].plot()

        # 顯示 FPS
        avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)
        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLOv11 Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_list:
        print(f"\n平均 FPS: {sum(fps_list) / len(fps_list):.1f}")


if __name__ == "__main__":
    main()
