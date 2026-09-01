@echo off
REM Windows 模型下载脚本
REM 自动下载所有语音模型到 D:\models\sherpa-onnx

setlocal enabledelayedexpansion
chcp 65001 > nul

set "MODEL_DIR=D:\models\sherpa-onnx"
set "PYTHON=python"

echo 语音模型下载工具
echo =====================================
echo 目标目录: %MODEL_DIR%
echo.

REM 检查 Python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo ✗ 未找到 Python，请先安装 Python
    exit /b 1
)
echo ✓ Python 已安装

REM 创建目录
if not exist "%MODEL_DIR%" (
    mkdir "%MODEL_DIR%"
    echo ✓ 创建目录: %MODEL_DIR%
)

REM 安装依赖
echo.
echo 安装依赖库...
%PYTHON% -m pip install huggingface-hub -q
%PYTHON% -m pip install sherpa-onnx sounddevice soundfile -q
echo ✓ 依赖库安装完成

REM 下载模型脚本
cat > "%MODEL_DIR%\download_models.py" << 'EOF'
#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path(__file__).parent

models = {
    "ASR中文模型": {
        "repo_id": "csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
        "local_dir": MODEL_DIR / "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
    },
    "TTS中文模型": {
        "repo_id": "espnet/piper-zh",
        "local_dir": MODEL_DIR / "piper-zh"
    },
    "KWS唤醒词模型": {
        "repo_id": "csukuangfj/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        "local_dir": MODEL_DIR / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    }
}

for name, info in models.items():
    print(f"\n下载 {name}...")
    print(f"Repository: {info['repo_id']}")
    
    try:
        snapshot_download(
            repo_id=info['repo_id'],
            local_dir=str(info['local_dir']),
            repo_type="model"
        )
        print(f"✓ {name} 下载完成: {info['local_dir']}")
    except Exception as e:
        print(f"✗ {name} 下载失败: {e}")

print("\n下载完成！")
print(f"模型目录: {MODEL_DIR}")
EOF

echo.
echo 现在下载模型（这需要一些时间，取决于网络速度）...
echo =====================================
%PYTHON% "%MODEL_DIR%\download_models.py"

if errorlevel 1 (
    echo.
    echo ✗ 模型下载失败
    echo.
    echo 备选方案：
    echo 1. 手动从 Hugging Face 下载：
    echo    - ASR: https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
    echo    - TTS: https://huggingface.co/espnet/piper-zh
    echo    - KWS: https://huggingface.co/csukuangfj/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
    echo.
    exit /b 1
)

echo.
echo ✓ 所有模型下载完成！
echo =====================================
echo.
echo 后续操作：
echo 1. 编辑 demo.py，确保 MODEL_BASE_DIR 指向 %MODEL_DIR%
echo 2. 运行: python demo.py
echo.
pause
