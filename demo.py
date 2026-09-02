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
    """演示：语音识别（使用 VAD 自动切分 + 预处理 + 回调低延迟）

    特性：
    - 使用 webrtcvad 进行语音活动检测（VAD）
    - 使用 sounddevice.InputStream 的回调获取低延迟音频块
    - 在识别前做简单预处理：去直流偏置（DC）、归一化（峰值归一）

    依赖：
    - pip install sounddevice webrtcvad
    """
    print("\n=== ASR (VAD + Preproc + Callback) 演示 ===")
    print("持续监听麦克风，检测到语音自动分段并识别。按 Ctrl+C 退出...")

    try:
        import sounddevice as sd
        import numpy as np
        import webrtcvad
        import queue
        import threading
        import time

        sample_rate = ASR_CONFIG.get("sample_rate", 16000)
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad 支持的采样率仅为 8000/16000/32000/48000")

        frame_ms = 30  # webrtcvad 支持 10/20/30 ms
        frame_size = int(sample_rate * frame_ms / 1000)  # 每帧样本数
        vad = webrtcvad.Vad(2)  # aggressiveness: 0-3，数字越大越激进

        asr = ASREngine(**ASR_CONFIG)

        audio_q = queue.Queue()
        stop_event = threading.Event()

        # 回调函数：把每个回调块放入队列
        def sd_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"InputStream status: {status}")
            # 确保是 int16，如果不是则转换
            # sounddevice 可配置 dtype='int16'，但如果设备不支持，会收到 float32
            if indata.dtype == np.int16:
                audio_q.put(indata.copy())
            else:
                # 将 float32 转为 int16
                clipped = np.clip(indata, -1.0, 1.0)
                int16 = (clipped * 32767).astype(np.int16)
                audio_q.put(int16)

        # 处理线程：从队列读取，组装为 frame_size 长度的帧并运行 VAD
        def processing_thread():
            buffer_frames = []  # 存储 int16 小块，待组装
            speech_buffer = []  # 存储检测到为语音的帧
            in_speech = False
            silence_frames = 0
            silence_timeout_ms = 300
            max_silence_frames = max(1, int(silence_timeout_ms / frame_ms))

            partial_chunk = np.empty((0,), dtype=np.int16)

            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.1)  # chunk 是 (N, channels) int16
                except queue.Empty:
                    continue

                # 单通道处理
                if chunk.ndim > 1:
                    chunk = chunk[:, 0]
                chunk = chunk.flatten()

                # 追加到部分缓存，并从中切出固定长度帧
                partial_chunk = np.concatenate((partial_chunk, chunk))

                while len(partial_chunk) >= frame_size:
                    frame = partial_chunk[:frame_size]
                    partial_chunk = partial_chunk[frame_size:]

                    # 预处理：去直流偏置 + 峰值归一化
                    # 去直流
                    frame_float = frame.astype("float32")
                    frame_float -= np.mean(frame_float)
                    # 峰值归一化（以避免除零）
                    max_abs = np.max(np.abs(frame_float))
                    if max_abs > 0:
                        frame_float = frame_float / max_abs
                    # 还原到 int16 以供 VAD 使用
                    proc_int16 = (frame_float * 32767).astype(np.int16)

                    pcm_bytes = proc_int16.tobytes()
                    is_speech = False
                    try:
                        is_speech = vad.is_speech(pcm_bytes, sample_rate)
                    except Exception as e:
                        logger.error(f"VAD 调用失败: {e}")

                    if is_speech:
                        speech_buffer.append(proc_int16)
                        in_speech = True
                        silence_frames = 0
                    else:
                        if in_speech:
                            silence_frames += 1
                            if silence_frames > max_silence_frames:
                                # 语音段结束，进行识别
                                if speech_buffer:
                                    segment_int16 = np.concatenate(speech_buffer)
                                    # 转为 float32，范围 [-1, 1)
                                    segment_float = segment_int16.astype('float32') / 32768.0

                                    # 可选：在这里执行更复杂的降噪/增强（占位）
                                    # e.g., spectral gating, Wiener filter, RNNoise, etc.

                                    try:
                                        result = asr.recognize(segment_float)
                                        print(f"\n识别结果: {result}\n")
                                    except Exception as e:
                                        logger.error(f"ASR 识别失败: {e}")

                                # 重置状态
                                speech_buffer = []
                                in_speech = False
                                silence_frames = 0
                        else:
                            # 非语音，且当前不在语音段
                            pass

            # 线程退出前若仍有未处理的语音缓存，可选择处理或丢弃
            if speech_buffer:
                try:
                    segment_int16 = np.concatenate(speech_buffer)
                    segment_float = segment_int16.astype('float32') / 32768.0
                    result = asr.recognize(segment_float)
                    print(f"\n识别结果: {result}\n")
                except Exception as e:
                    logger.error(f"ASR 识别失败(结束时): {e}")

        # 启动输入流与处理线程
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=frame_size, callback=sd_callback)
            thread = threading.Thread(target=processing_thread, daemon=True)

            thread.start()
            with stream:
                print("开始监听... (按 Ctrl+C 停止)")
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n已停止监听。")
        finally:
            stop_event.set()
            # 清空队列以释放资源
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except Exception:
                    break

    except Exception as e:
        logger.error(f"ASR(VAD) 演示失败: {e}")


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
