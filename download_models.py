#!/usr/bin/env python3
"""
跨平台模型下载脚本
支持 Windows、Linux、macOS
"""

import os
import sys
from pathlib import Path

# 尝试导入依赖
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("⚠️  未安装 huggingface-hub，正在安装...")
    os.system(f"{sys.executable} -m pip install huggingface-hub -q")
    from huggingface_hub import snapshot_download


def download_models(model_dir: str = None):
    """下载所有模型"""
    
    # 确定模型目录
    if model_dir is None:
        if sys.platform == "win32":
            model_dir = "D:/models/sherpa-onnx"
        else:
            model_dir = os.path.expanduser("~/models/sherpa-onnx")
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("语音模型下载工具")
    print("=" * 60)
    print(f"目标目录: {model_dir}")
    print()
    
    models = {
        "ASR 中文模型（语音识别）": {
            "repo_id": "csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
            "local_dir": model_dir / "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
        },
        "TTS 中文模型（语音合成）": {
            "repo_id": "espnet/piper-zh",
            "local_dir": model_dir / "piper-zh"
        },
        "KWS 唤醒词模型（唤醒词检测）": {
            "repo_id": "csukuangfj/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
            "local_dir": model_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        }
    }
    
    success_count = 0
    fail_count = 0
    
    for idx, (name, info) in enumerate(models.items(), 1):
        print(f"[{idx}/{len(models)}] 下载 {name}...")
        print(f"     Repository: {info['repo_id']}")
        
        # 如果已经存在，跳过
        if info['local_dir'].exists():
            print(f"     ✓ 已存在，跳过")
            success_count += 1
            continue
        
        try:
            snapshot_download(
                repo_id=info['repo_id'],
                local_dir=str(info['local_dir']),
                repo_type="model",
                resume_download=True,
            )
            print(f"     ✓ 下载完成: {info['local_dir']}")
            success_count += 1
        except Exception as e:
            print(f"     ✗ 下载失败: {e}")
            fail_count += 1
        
        print()
    
    # 总结
    print("=" * 60)
    print(f"下载完成: 成功 {success_count}/{len(models)}")
    if fail_count > 0:
        print(f"         失败 {fail_count}/{len(models)}")
    print("=" * 60)
    
    # 显示模型配置
    print("\n模型配置（复制到 demo.py 的 MODEL_BASE_DIR）:")
    print(f"MODEL_BASE_DIR = r\"{model_dir}\"")
    
    # 验证模型完整性
    print("\n验证模型文件:")
    all_ok = True
    for name, info in models.items():
        local_dir = info['local_dir']
        if local_dir.exists():
            onnx_files = list(local_dir.glob("*.onnx"))
            txt_files = list(local_dir.glob("*.txt"))
            dict_files = list(local_dir.glob("*.dict"))
            
            total = len(onnx_files) + len(txt_files) + len(dict_files)
            if total > 0:
                print(f"  ✓ {name}: {total} 个文件")
            else:
                print(f"  ✗ {name}: 无文件")
                all_ok = False
        else:
            print(f"  ✗ {name}: 目录不存在")
            all_ok = False
    
    if all_ok:
        print("\n✓ 所有模型都已正确下载！")
        print("\n下一步:")
        print("1. 修改 demo.py 中的 MODEL_BASE_DIR")
        print("2. 运行: python demo.py")
        return 0
    else:
        print("\n✗ 某些模型下载失败或不完整")
        return 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载语音模型")
    parser.add_argument("--model-dir", default=None, 
                       help="模型保存目录（默认: Windows D:/models/sherpa-onnx, 其他系统 ~/models/sherpa-onnx）")
    
    args = parser.parse_args()
    
    sys.exit(download_models(args.model_dir))
