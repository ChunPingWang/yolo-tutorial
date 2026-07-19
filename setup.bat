@echo off
REM YOLOv11 環境安裝腳本 - Windows + RTX 4060
echo ====================================
echo  YOLOv11 環境安裝
echo ====================================

REM 檢查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 未找到 Python，請先安裝 Python 3.10+
    exit /b 1
)

REM 檢查 NVIDIA Driver
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [錯誤] 未找到 NVIDIA Driver，請先安裝驅動程式
    exit /b 1
)

REM 建立虛擬環境
if not exist "venv" (
    echo.
    echo [0/5] 建立虛擬環境...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo.
echo [1/5] 安裝 PyTorch (CUDA 12.x)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/5] 安裝 Ultralytics (YOLOv11)...
pip install ultralytics

echo.
echo [3/5] 安裝相依套件...
pip install opencv-python onnx onnxruntime-gpu

echo.
echo [4/5] 安裝 TensorRT (CUDA 12.x)...
pip install tensorrt==10.9.0.34

echo.
echo [5/5] 驗證 CUDA...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

echo.
echo ====================================
echo  安裝完成！
echo ====================================
pause
