#!/usr/bin/env python3
"""
语音系统演示脚本
展示如何使用ASR、TTS、KWS模块
"""

import os
from pathlib import Path
from speech_system import ASREngine, TTSEngine, KWSEngine, SpeechRecognitionSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 配置模型路径 ====================
# 根据你的模型下载位置修改这个路径
MODEL_BASE_DIR = "D:/models/sherpa-onnx"  # Windows 示例路径
# MODEL_BASE_DIR = "/home/user/models/sherpa-onnx"  # Linux 示例路径

# ASR 模型配置
ASR_CONFIG = {
    "tokens": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt"),
    "encoder": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/encoder.onnx"),
    "decoder": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/decoder.onnx"),
    "joiner": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/joiner.onnx"),
    "sample_rate": 16000,
    "provider": "cpu",  # 改为 "cuda" 使用GPU
}

# TTS 模型配置
TTS_CONFIG = {
    "vits_model": os.path.join(MODEL_BASE_DIR, "piper-zh/zh_CN-huayan-medium.onnx"),
    "vits_lexicon": os.path.join(MODEL_BASE_DIR, "piper-zh/zh_CN.dict"),
    "vits_tokens": os.path.join(MODEL_BASE_DIR, "piper-zh/tokens.txt"),
    "vits_data_dir": os.path.join(MODEL_BASE_DIR, "piper-zh"),
    "provider": "cpu",
}

# KWS 模型配置
KWS_CONFIG = {
    "tokens": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"),
    "encoder": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder.onnx"),
    "decoder": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder.onnx"),
    "joiner": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner.onnx"),
    "keywords_file": os.path.join(MODEL_BASE_DIR, "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/keywords.txt"),
    "provider": "cpu",
}


def demo_asr():
    """演示：语音识别"""
    print("\n=== ASR 演示 ===")
    print("确保麦克风可用，3秒内说话...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        asr = ASREngine(**ASR_CONFIG)
        
        # 从麦克风录音
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        
        # 识别
        result = asr.recognize(audio)
        print(f"识别结果: {result}")
    
    except Exception as e:
        logger.error(f"ASR演示失败: {e}")


def demo_tts():
    """演示：语音合成"""
    print("\n=== TTS 演示 ===")
    
    try:
        tts = TTSEngine(**TTS_CONFIG)
        
        # 生成语音并播放
        text = "你好，我是语音助手。"
        print(f"合成文本: {text}")
        tts.speak(text)
        
        # 也可以保存到文件
        output_file = "output.wav"
        tts.save_audio(text, output_file)
        print(f"音频已保存: {output_file}")
    
    except Exception as e:
        logger.error(f"TTS演示失败: {e}")


def demo_kws():
    """演示：唤醒词检测"""
    print("\n=== KWS 演示 ===")
    print("请说: '你好军哥' 或其他唤醒词...")
    print("3秒内说话...")
    
    try:
        import sounddevice as sd
        
        kws = KWSEngine(**KWS_CONFIG)
        
        # 从麦克风录音
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        
        # 检测唤醒词
        result = kws.detect(audio)
        if result:
            print(f"检测到唤醒词: {result}")
        else:
            print("未检测到唤醒词")
    
    except Exception as e:
        logger.error(f"KWS演示失败: {e}")


def demo_interactive():
    """演示：交互模式"""
    print("\n=== 交互模式演示 ===")
    print("说话 -> 识别 -> 回复 (Ctrl+C退出)")
    
    try:
        system = SpeechRecognitionSystem(
            asr_config=ASR_CONFIG,
            tts_config=TTS_CONFIG,
        )
        system.interactive_mode()
    
    except Exception as e:
        logger.error(f"交互演示失败: {e}")


def check_models():
    """检查模型文件是否存在"""
    print("\n=== 检查模型文件 ===")
    
    all_configs = {
        "ASR": ASR_CONFIG,
        "TTS": TTS_CONFIG,
        "KWS": KWS_CONFIG,
    }
    
    all_exist = True
    for name, config in all_configs.items():
        print(f"\n{name} 配置:")
        for key, path in config.items():
            if isinstance(path, str) and path.endswith(('.onnx', '.txt', '.dict')):
                exists = Path(path).exists()
                status = "✓" if exists else "✗"
                print(f"  {status} {key}: {path}")
                if not exists:
                    all_exist = False
    
    if not all_exist:
        print("\n⚠️ 某些模型文件不存在，请先下载模型")
    else:
        print("\n✓ 所有模型文件都存在")


if __name__ == "__main__":
    print("语音系统演示")
    print("=" * 50)
    
    # 检查模型
    check_models()
    
    # 选择演示
    print("\n选择演示:")
    print("1. ASR (语音识别)")
    print("2. TTS (语音合成)")
    print("3. KWS (唤醒词检测)")
    print("4. 交互模式")
    print("5. 退出")
    
    choice = input("\n请输入选项 (1-5): ").strip()
    
    if choice == "1":
        demo_asr()
    elif choice == "2":
        demo_tts()
    elif choice == "3":
        demo_kws()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        print("退出")
    else:
        print("无效选项")