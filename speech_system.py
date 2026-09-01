#!/usr/bin/env python3
"""
独立的语音系统 - 无需ROS，支持Windows
包括：ASR（语音识别）、TTS（语音合成）、KWS（唤醒词检测）
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Callable
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
except ImportError:
    logger.error("Please install sherpa-onnx: pip install sherpa-onnx")
    sys.exit(1)


class ASREngine:
    """语音识别引擎（Speech-to-Text）"""
    
    def __init__(self, 
                 tokens: str,
                 encoder: str,
                 decoder: str,
                 joiner: str,
                 sample_rate: int = 16000,
                 provider: str = "cpu"):
        """
        初始化ASR引擎
        
        Args:
            tokens: tokens.txt 文件路径
            encoder: encoder.onnx 文件路径
            decoder: decoder.onnx 文件路径
            joiner: joiner.onnx 文件路径
            sample_rate: 采样率，默认16000Hz
            provider: 计算提供者，"cpu" 或 "cuda"
        """
        self.sample_rate = sample_rate
        logger.info("初始化ASR引擎...")
        
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=1,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
            decoding_method="greedy_search",
            provider=provider,
        )
        self.stream = self.recognizer.create_stream()
        logger.info("ASR引擎初始化完成")
    
    def recognize(self, audio_data: np.ndarray) -> str:
        """
        识别音频
        
        Args:
            audio_data: numpy 数组，float32 格式
            
        Returns:
            识别结果文本
        """
        self.stream.accept_waveform(self.sample_rate, audio_data)
        
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        
        result = ""
        if self.recognizer.is_endpoint(self.stream):
            result = self.recognizer.get_result(self.stream)
            self.recognizer.reset(self.stream)
        
        return result
    
    def recognize_from_file(self, audio_file: str) -> str:
        """从音频文件识别"""
        audio, sr = sf.read(audio_file, dtype='float32')
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        if sr != self.sample_rate:
            logger.warning(f"采样率不匹配: {sr} vs {self.sample_rate}")
        
        return self.recognize(audio)


class TTSEngine:
    """语音合成引擎（Text-to-Speech）"""
    
    def __init__(self,
                 vits_model: str,
                 vits_lexicon: str,
                 vits_tokens: str,
                 vits_data_dir: str = "",
                 vits_dict_dir: str = "",
                 tts_rule_fsts: str = "",
                 provider: str = "cpu",
                 num_threads: int = 1):
        """
        初始化TTS引擎
        
        Args:
            vits_model: VITS 模型文件路径
            vits_lexicon: 词典文件路径
            vits_tokens: tokens 文件路径
            vits_data_dir: 数据目录
            vits_dict_dir: 字典目录
            tts_rule_fsts: FST 规则文件
            provider: 计算提供者，"cpu" 或 "cuda"
            num_threads: 线程数
        """
        logger.info("初始化TTS引擎...")
        
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=vits_model,
                    lexicon=vits_lexicon,
                    data_dir=vits_data_dir,
                    dict_dir=vits_dict_dir,
                    tokens=vits_tokens,
                ),
                provider=provider,
                debug=False,
                num_threads=num_threads,
            ),
            rule_fsts=tts_rule_fsts,
            max_num_sentences=1,
        )
        
        if not tts_config.validate():
            logger.error("TTS配置无效")
            sys.exit(1)
        
        self.tts = sherpa_onnx.OfflineTts(tts_config)
        self.sample_rate = self.tts.sample_rate
        logger.info(f"TTS引擎初始化完成，采样率：{self.sample_rate}Hz")
    
    def synthesize(self, text: str, sid: int = 0, speed: float = 1.0) -> np.ndarray:
        """
        文本转语音
        
        Args:
            text: 输入文本
            sid: 说话人ID，默认0
            speed: 语速，默认1.0
            
        Returns:
            音频数据（numpy数组）
        """
        logger.info(f"生成语音: '{text}'")
        start_time = time.time()
        
        audio = self.tts.generate(text, sid=sid, speed=speed)
        
        elapsed = time.time() - start_time
        duration = len(audio.samples) / self.sample_rate
        logger.info(f"生成耗时: {elapsed:.3f}秒, 音频时长: {duration:.3f}秒")
        
        return audio.samples
    
    def speak(self, text: str, sid: int = 0, speed: float = 1.0):
        """生成并播放语音"""
        audio_data = self.synthesize(text, sid, speed)
        logger.info("播放音频...")
        sd.play(audio_data, samplerate=self.sample_rate)
        sd.wait()
        logger.info("播放完成")
    
    def save_audio(self, text: str, output_file: str, sid: int = 0, speed: float = 1.0):
        """生成语音并保存到文件"""
        audio_data = self.synthesize(text, sid, speed)
        sf.write(output_file, audio_data, samplerate=self.sample_rate, subtype="PCM_16")
        logger.info(f"音频已保存: {output_file}")


class KWSEngine:
    """唤醒词检测引擎（Keyword Spotting）"""
    
    def __init__(self,
                 tokens: str,
                 encoder: str,
                 decoder: str,
                 joiner: str,
                 keywords_file: str,
                 sample_rate: int = 16000,
                 provider: str = "cpu",
                 keywords_threshold: float = 0.25):
        """
        初始化KWS引擎
        
        Args:
            tokens: tokens.txt 文件路径
            encoder: encoder.onnx 文件路径
            decoder: decoder.onnx 文件路径
            joiner: joiner.onnx 文件路径
            keywords_file: 唤醒词文件路径
            sample_rate: 采样率，默认16000Hz
            provider: 计算提供者，"cpu" 或 "cuda"
            keywords_threshold: 唤醒词检测阈值
        """
        self.sample_rate = sample_rate
        logger.info("初始化KWS引擎...")
        
        self.keyword_spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=1,
            max_active_paths=4,
            keywords_file=keywords_file,
            keywords_score=1.0,
            keywords_threshold=keywords_threshold,
            num_trailing_blanks=1,
            provider=provider,
        )
        self.stream = self.keyword_spotter.create_stream()
        logger.info("KWS引擎初始化完成")
    
    def detect(self, audio_data: np.ndarray) -> Optional[str]:
        """
        检测唤醒词
        
        Args:
            audio_data: numpy 数组，float32 格式
            
        Returns:
            检测到的唤醒词，未检测到返回 None
        """
        self.stream.accept_waveform(self.sample_rate, audio_data)
        
        while self.keyword_spotter.is_ready(self.stream):
            self.keyword_spotter.decode_stream(self.stream)
        
        result = self.keyword_spotter.get_result(self.stream)
        return result if result else None
    
    def detect_from_file(self, audio_file: str) -> Optional[str]:
        """从音频文件检测唤醒词"""
        audio, sr = sf.read(audio_file, dtype='float32')
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        return self.detect(audio)


class SpeechRecognitionSystem:
    """完整的语音系统"""
    
    def __init__(self,
                 asr_config: Optional[dict] = None,
                 tts_config: Optional[dict] = None,
                 kws_config: Optional[dict] = None):
        """
        初始化语音系统
        
        Args:
            asr_config: ASR配置字典
            tts_config: TTS配置字典
            kws_config: KWS配置字典
        """
        self.asr = None
        self.tts = None
        self.kws = None
        
        if asr_config:
            self.asr = ASREngine(**asr_config)
        
        if tts_config:
            self.tts = TTSEngine(**tts_config)
        
        if kws_config:
            self.kws = KWSEngine(**kws_config)
    
    def interactive_mode(self):
        """交互模式：麦克风输入 -> 识别 -> 合成回复 -> 播放"""
        if not self.asr or not self.tts:
            logger.error("需要同时配置ASR和TTS")
            return
        
        logger.info("进入交互模式（按Ctrl+C退出）...")
        logger.info("开始录音，说话5秒后自动停止...")
        
        try:
            while True:
                # 录音
                audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
                sd.wait()
                audio = audio.flatten()
                
                # 识别
                text = self.asr.recognize(audio)
                if text:
                    logger.info(f"识别结果: {text}")
                    
                    # 生成回复
                    reply = f"你说了：{text}"
                    self.tts.speak(reply)
                else:
                    logger.warning("未识别到语音")
        
        except KeyboardInterrupt:
            logger.info("退出交互模式")


if __name__ == "__main__":
    print("Speech System Module - Import this in your scripts")