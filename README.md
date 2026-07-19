# YOLOv11 物件偵測教學 — Windows + RTX 4060 Laptop GPU

本教學以 **YOLOv11** 為核心，示範如何在 Windows 筆電搭配 NVIDIA RTX 4060 Laptop GPU 上完成物件偵測，並透過 TensorRT 加速推論。

---

## 硬體與軟體環境

| 項目 | 規格 |
|------|------|
| CPU | Intel Core i7-13620H (13th Gen, 10 核心) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6) |
| RAM | 48 GB DDR5 |
| OS | Windows 11 Enterprise |
| CUDA | 12.8 |
| Driver | 571.96 |

> 💡 相較於 Jetson Orin Nano 的 8GB 共享記憶體架構，本機 GPU 擁有獨立 8GB VRAM 且 CPU 端有 48GB RAM，適合進行較大模型的推論與訓練實驗。

---

## 基礎知識

### 什麼是 YOLO？

YOLO（You Only Look Once）是一種單階段物件偵測模型，能在一次前向傳播中同時預測邊界框與類別。YOLOv11 是 Ultralytics 發布的最新版本，採用 anchor-free 架構，支援偵測、分割、姿態估計等多種任務。

### 電腦視覺三大任務

| 任務 | 說明 | 輸出 |
|------|------|------|
| 分類 (Classification) | 判斷圖片中的主要物體 | 類別標籤 |
| 偵測 (Detection) | 定位所有物體的位置與類別 | 邊界框 + 類別 |
| 分割 (Segmentation) | 像素級別標注物體輪廓 | 遮罩 (mask) |

### 推論流程

```
輸入影像 → 前處理 (Resize + Normalize) → 模型推論 → 後處理 (NMS) → 視覺化輸出
```

### TensorRT 加速原理

TensorRT 是 NVIDIA 的推論優化引擎，透過以下方式加速：
- **層融合 (Layer Fusion)**：合併多個運算為單一 kernel
- **精度量化 (Quantization)**：FP32 → FP16 / INT8 降低計算量
- **記憶體優化**：減少中間結果的記憶體搬移
- **硬體特化**：針對特定 GPU 架構生成最佳化的執行計畫

> ⚠️ TensorRT engine 與硬體綁定，無法跨不同 GPU 使用。

---

## 教學步驟總覽

| 步驟 | 內容 | 腳本 |
|------|------|------|
| Step 1 | 環境安裝 | `setup.bat` |
| Step 2 | 第一張圖片偵測 | `src/detect_image.py` |
| Step 3 | 理解偵測結果 | — |
| Step 4 | 即時攝影機偵測 | `src/detect_camera.py` |
| Step 5 | TensorRT 加速 | `src/export_tensorrt.py` |
| Step 6 | 效能基準測試 | `src/benchmark.py` |

---

## Step 1：環境安裝

### 前置需求

- Python 3.10+ (建議使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- NVIDIA Driver 已安裝 (本機: 571.96)
- Git

### 安裝步驟

```bash
# 建立虛擬環境
conda create -n yolo python=3.10 -y
conda activate yolo

# 安裝 PyTorch (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝 Ultralytics (含 YOLOv11)
pip install ultralytics

# 安裝其他相依套件
pip install opencv-python onnx onnxruntime-gpu

# 安裝 TensorRT (CUDA 12.x)
pip install tensorrt==10.9.0.34

# 驗證 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

預期輸出：
```
CUDA available: True
Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Step 2：第一張圖片偵測

執行你的第一次物件偵測：

```bash
python src/detect_image.py --source https://ultralytics.com/images/bus.jpg --save-json
```

此腳本會：
1. 自動下載 `yolo11n.pt` 模型 (≈5.4 MB)
2. 對輸入圖片執行推論
3. 將偵測結果儲存為 JSON 並輸出標注圖片至 `output/`

---

## Step 3：理解偵測結果

JSON 輸出結構：

```json
{
  "file": "bus.jpg",
  "image_size": [1080, 810],
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox_xyxy": [17, 231, 801, 768]
    }
  ]
}
```

欄位說明：
- `class`：偵測到的物體類別 (COCO 80 類)
- `confidence`：信心分數 (0~1)
- `bbox_xyxy`：邊界框座標 [x1, y1, x2, y2]

---

## Step 4：即時攝影機偵測

```bash
# 使用預設攝影機 (USB Webcam / 筆電內建鏡頭)
python src/detect_camera.py

# 指定攝影機 ID
python src/detect_camera.py --source 0
```

在 Windows 環境下使用 DirectShow 後端，按 `q` 鍵退出。

---

## Step 5：TensorRT 加速

將 PyTorch 模型轉換為 TensorRT engine：

```bash
# FP16 精度 (推薦，速度與精度兼顧)
python src/export_tensorrt.py --weights models/yolo11n.pt --half

# INT8 精度 (最快，需要校準資料)
python src/export_tensorrt.py --weights models/yolo11n.pt --int8 --data coco128.yaml
```

轉換流程：`.pt` → `.onnx` → `.engine`

> RTX 4060 支援 FP16 與 INT8 Tensor Core 加速，轉換時間約 2-5 分鐘。

---

## Step 6：效能基準測試

```bash
python src/benchmark.py --weights models/yolo11n.pt --warmup 20 --iters 200
```

輸出指標：
- 平均延遲 (ms)
- P50 / P95 / P99 延遲
- 吞吐量 (FPS)

### RTX 4060 Laptop GPU 實測效能

| 模型 | 格式 | FPS | 平均延遲 (ms) | P95 延遲 (ms) |
|------|------|-----|--------------|--------------|
| yolo11n | PyTorch FP32 | 113.7 | 8.80 | 11.31 |
| yolo11n | TensorRT FP16 | 214.0 | 4.67 | 5.47 |

> 實測環境：Windows 11, Driver 571.96, CUDA 12.8, TensorRT 10.9.0, 640x640 輸入。

---

## 模型選擇指南

| 需求 | 推薦模型 | 格式 |
|------|----------|------|
| 最高 FPS | yolo11n | TensorRT INT8 |
| 速度精度兼顧 | yolo11n | TensorRT FP16 |
| 最高精度 | yolo11l | TensorRT FP16 |
| 開發除錯 | yolo11n | PyTorch FP32 |

---

## 疑難排解

| 問題 | 解決方式 |
|------|----------|
| `torch.cuda.is_available()` 回傳 False | 確認已安裝 CUDA 版本的 PyTorch，非 CPU 版本 |
| TensorRT 轉換失敗 | 確認已安裝 `tensorrt` 套件，版本需與 CUDA 相容 |
| 攝影機無法開啟 | 確認無其他程式佔用攝影機；嘗試不同 `--source` ID |
| GPU 記憶體不足 | 使用較小模型 (yolo11n) 或降低輸入解析度 |
| Windows Defender 影響效能 | 將工作目錄加入排除清單 |

---

## 專案結構

```
yolo-tutorial/
├── README.md           # 本教學文件
├── setup.bat           # Windows 環境安裝腳本
├── src/
│   ├── detect_image.py     # 圖片偵測
│   ├── detect_camera.py    # 攝影機即時偵測
│   ├── export_tensorrt.py  # TensorRT 轉換
│   └── benchmark.py        # 效能測試
├── models/             # 模型權重
├── data/               # 測試圖片
└── output/             # 輸出結果
```

---

## 參考資源

- [Ultralytics YOLOv11 文件](https://docs.ultralytics.com/)
- [NVIDIA TensorRT 文件](https://developer.nvidia.com/tensorrt)
- [原始 Jetson 版教學](https://github.com/ChunPingWang/yolo-tutorial-for-jetson)

---

## 授權

MIT License
