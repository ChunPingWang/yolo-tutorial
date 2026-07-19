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

echo.
echo [1/4] 安裝 PyTorch (CUDA 12.x)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [2/4] 安裝 Ultralytics (YOLOv11)...
pip install ultralytics

echo.
echo [3/4] 安裝相依套件...
pip install opencv-python onnx onnxruntime-gpu

echo.
echo [4/4] 驗證 CUDA...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

echo.
echo ====================================
echo  安裝完成！
echo ====================================
pause
