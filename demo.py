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


def reduce_noise(signal: "np.ndarray", sr: int):
    """尝试多种降噪方法：
    - 优先使用 noisereduce.reduce_noise（依赖 noisereduce）
    - 其次使用 scipy.signal.wiener
    - 否则使用简单的静默段均值去除作为退化方案

    输入：float32, 范围大约 [-1, 1]
    返回：float32
    """
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=signal, sr=sr)
    except Exception:
        try:
            from scipy.signal import wiener
            # wiener 有时对语音效果有限，但作为快速 fallback 是可用的
            den = wiener(signal)
            return den.astype("float32")
        except Exception:
            # 最简单的降噪：以前 0.2s 作为噪声估计，减去其均值
            import numpy as np
            if len(signal) < max(1, int(0.2 * sr)):
                return signal
            noise_est = signal[: int(0.2 * sr)]
            noise_mean = np.mean(noise_est)
            return (signal - noise_mean).astype("float32")


def demo_asr():
    """演示：语音识别（使用 VAD 自动切分 + 预处理 + 回调低延迟）

    特性：
    - 使用 webrtcvad 进行语音活动检测（VAD）
    - 使用 sounddevice.InputStream 的回调获取��延迟音频块
    - 在识别前做简单预处理：去直流偏置（DC）、峰值归一
    - 提供更强的降噪入口（reduce_noise）

    依赖：
    - pip install sounddevice webrtcvad numpy
    - 可选：noisereduce / scipy（用于更好降噪）
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
            if indata.dtype == np.int16:
                audio_q.put(indata.copy())
            else:
                clipped = np.clip(indata, -1.0, 1.0)
                int16 = (clipped * 32767).astype(np.int16)
                audio_q.put(int16)

        # 处理线程：从队列读取，组装为 frame_size 长度的帧并运行 VAD
        def processing_thread():
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
                    frame_float = frame.astype("float32")
                    frame_float -= np.mean(frame_float)
                    max_abs = np.max(np.abs(frame_float))
                    if max_abs > 0:
                        frame_float = frame_float / max_abs
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
                                    segment_float = segment_int16.astype("float32") / 32768.0

                                    # 降噪（尝试多种方法）
                                    try:
                                        denoised = reduce_noise(segment_float, sample_rate)
                                    except Exception as e:
                                        logger.debug(f"降噪失败，使用原始音频: {e}")
                                        denoised = segment_float

                                    try:
                                        result = asr.recognize(denoised)
                                        print(f"\n识别结果: {result}\n")
                                    except Exception as e:
                                        logger.error(f"ASR 识别失败: {e}")

                                # 重置状态
                                speech_buffer = []
                                in_speech = False
                                silence_frames = 0
                        else:
                            pass

            # 线程退出前若仍有未处理的语音缓存，可选择处理或丢弃
            if speech_buffer:
                try:
                    segment_int16 = np.concatenate(speech_buffer)
                    segment_float = segment_int16.astype('float32') / 32768.0
                    denoised = reduce_noise(segment_float, sample_rate)
                    result = asr.recognize(denoised)
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
    """演示：唤醒词检测（使用 VAD + 回调 + 可选降噪）

    与 demo_asr 类似的低延迟实现，持续监听麦克风并对检测到的语音段调用 KWSEngine.detect。
    检测到唤醒词会打印并继续监听。
    """
    print("\n=== KWS 演示 (VAD + Callback) ===")
    print("持续监听麦克风以检测唤醒词（按 Ctrl+C 退出）")

    try:
        import sounddevice as sd
        import numpy as np
        import webrtcvad
        import queue
        import threading
        import time

        sample_rate = KWS_CONFIG.get("sample_rate", ASR_CONFIG.get("sample_rate", 16000))
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad 支持的采样率仅为 8000/16000/32000/48000")

        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)
        vad = webrtcvad.Vad(1)  # 对唤醒词检测用较保守的阈值

        kws = KWSEngine(**KWS_CONFIG)

        audio_q = queue.Queue()
        stop_event = threading.Event()

        def sd_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"KWS InputStream status: {status}")
            if indata.dtype == np.int16:
                audio_q.put(indata.copy())
            else:
                clipped = np.clip(indata, -1.0, 1.0)
                int16 = (clipped * 32767).astype(np.int16)
                audio_q.put(int16)

        def processing_thread():
            speech_buffer = []
            in_speech = False
            silence_frames = 0
            silence_timeout_ms = 300
            max_silence_frames = max(1, int(silence_timeout_ms / frame_ms))

            partial_chunk = np.empty((0,), dtype=np.int16)

            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk.ndim > 1:
                    chunk = chunk[:, 0]
                chunk = chunk.flatten()

                partial_chunk = np.concatenate((partial_chunk, chunk))

                while len(partial_chunk) >= frame_size:
                    frame = partial_chunk[:frame_size]
                    partial_chunk = partial_chunk[frame_size:]

                    # 预处理
                    frame_float = frame.astype("float32")
                    frame_float -= np.mean(frame_float)
                    max_abs = np.max(np.abs(frame_float))
                    if max_abs > 0:
                        frame_float = frame_float / max_abs
                    proc_int16 = (frame_float * 32767).astype(np.int16)

                    pcm_bytes = proc_int16.tobytes()
                    is_speech = False
                    try:
                        is_speech = vad.is_speech(pcm_bytes, sample_rate)
                    except Exception as e:
                        logger.error(f"KWS VAD 调用失败: {e}")

                    if is_speech:
                        speech_buffer.append(proc_int16)
                        in_speech = True
                        silence_frames = 0
                    else:
                        if in_speech:
                            silence_frames += 1
                            if silence_frames > max_silence_frames:
                                if speech_buffer:
                                    segment_int16 = np.concatenate(speech_buffer)
                                    segment_float = segment_int16.astype("float32") / 32768.0

                                    # 对唤醒词检测先做降噪再检测
                                    try:
                                        denoised = reduce_noise(segment_float, sample_rate)
                                    except Exception as e:
                                        logger.debug(f"KWS 降噪失败，使用原始音频: {e}")
                                        denoised = segment_float

                                    try:
                                        result = kws.detect(denoised)
                                        if result:
                                            print(f"检测到唤醒词: {result}")
                                    except Exception as e:
                                        logger.error(f"KWS 检测失败: {e}")

                                speech_buffer = []
                                in_speech = False
                                silence_frames = 0
                        else:
                            pass

            # 退出前处理残留
            if speech_buffer:
                try:
                    segment_int16 = np.concatenate(speech_buffer)
                    segment_float = segment_int16.astype('float32') / 32768.0
                    denoised = reduce_noise(segment_float, sample_rate)
                    result = kws.detect(denoised)
                    if result:
                        print(f"检测到唤醒词: {result}")
                except Exception as e:
                    logger.error(f"KWS 检测失败(结束时): {e}")

        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=frame_size, callback=sd_callback)
            thread = threading.Thread(target=processing_thread, daemon=True)

            thread.start()
            with stream:
                print("KWS 开始监听... (按 Ctrl+C 停止)")
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nKWS 已停止监听。")
        finally:
            stop_event.set()
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except Exception:
                    break

    except Exception as e:
        logger.error(f"KWS 演示失败: {e}")


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
