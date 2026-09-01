
# VLM-Speech-Windows

独立的语音系统 - 无需 ROS，完全支持 Windows、Linux、macOS

包括三个核心模块：
- **ASR** (语音识别 Speech-to-Text)
- **TTS** (语音合成 Text-to-Speech)  
- **KWS** (唤醒词检测 Keyword Spotting)

所有模型都在本地运行，完全离线工作。

---

## 快速开始

### 1️⃣ 安装依赖

```bash
# 克隆项目
git clone https://github.com/daimou03/VLM-Speech-Windows.git
cd VLM-Speech-Windows

# 安装 Python 包
pip install -r requirements.txt
```

### 2️⃣ 下载模型

#### 方法 A：自动下载（推荐）

**Windows:**
```bash
python download_models.py
```

**Linux/macOS:**
```bash
python download_models.py --model-dir ~/models/sherpa-onnx
```

#### 方法 B：手动下载

从 Hugging Face 下载模型到本地：

| 模型 | 用途 | 下载 | 大小 |
|------|------|------|------|
| **sherpa-onnx-streaming-zipformer-zh** | ASR 语音识别 | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23) | ~50MB |
| **piper-zh** | TTS 语音合成 | [HF](https://huggingface.co/espnet/piper-zh) | ~200MB |
| **sherpa-onnx-kws** | KWS 唤醒词 | [HF](https://huggingface.co/csukuangfj/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01) | ~20MB |

下载后放到同一目录：
```
D:\models\sherpa-onnx\
├── sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23\
├── piper-zh\
└── sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01\
```

### 3️⃣ 修改配置

编辑 `demo.py`：

```python
MODEL_BASE_DIR = "D:/models/sherpa-onnx"  # 改为你的模型目录
```

### 4️⃣ 运行演示

```bash
python demo.py
```

选择功能：
- 1: ASR 语音识别
- 2: TTS 语音合成
- 3: KWS 唤醒词检测
- 4: 交互模式

---

## 使用示例

### 语音识别 (ASR)

```python
from speech_system import ASREngine
import sounddevice as sd

asr = ASREngine(
    tokens="path/to/tokens.txt",
    encoder="path/to/encoder.onnx",
    decoder="path/to/decoder.onnx",
    joiner="path/to/joiner.onnx",
)

# 从麦克风录音3秒
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

# 识别
text = asr.recognize(audio.flatten())
print(f"识别结果: {text}")
```

### 语音合成 (TTS)

```python
from speech_system import TTSEngine

tts = TTSEngine(
    vits_model="path/to/vits.onnx",
    vits_lexicon="path/to/lexicon.txt",
    vits_tokens="path/to/tokens.txt",
)

# 播放
tts.speak("你好，我是语音助手")

# 保存
tts.save_audio("你好", "output.wav")
```

### 唤醒词检测 (KWS)

```python
from speech_system import KWSEngine

kws = KWSEngine(
    tokens="path/to/tokens.txt",
    encoder="path/to/encoder.onnx",
    decoder="path/to/decoder.onnx",
    joiner="path/to/joiner.onnx",
    keywords_file="path/to/keywords.txt",
)

# 检测
result = kws.detect(audio.flatten())
if result:
    print(f"检测到唤醒词: {result}")
```

### 完整系统

```python
from speech_system import SpeechRecognitionSystem

system = SpeechRecognitionSystem(
    asr_config=ASR_CONFIG,
    tts_config=TTS_CONFIG,
)

# 交互模式
system.interactive_mode()
```

---

## 系统要求

- Python 3.8+
- Windows 10+、Linux、macOS
- 4GB 内存，20GB 磁盘（用于模型）

### 性能指标

| 硬件 | ASR RTF | TTS RTF |
|------|---------|---------|
| CPU | 0.3-0.5 | 0.2-0.4 |
| GPU | 0.02-0.05 | 0.01-0.02 |

（RTF < 1 = 实时处理）

---

## GPU 加速

### 使用 NVIDIA GPU

1. 安装 CUDA 工具包
2. 修改配置：

```python
ASR_CONFIG = {
    ...
    "provider": "cuda",
}

TTS_CONFIG = {
    ...
    "provider": "cuda",
}
```

3. 重新运行

---

## 模型信息

| 模型 | 语言 | 特点 |
|------|------|------|
| **ASR** | 中文 | 流式实时识别，低延迟 |
| **TTS** | 中文 | 自然流畅女性声音 |
| **KWS** | 中文 | 实时唤醒词检测 |

---

## 自定义唤醒词

编辑 `keywords.txt`：

```
你好军哥
唤醒小爱
自定义唤醒词
```

---

## 常见问题

**Q: 支持其他语言吗？**
当前只支持中文，可从 Hugging Face 下载其他语言模型

**Q: 可以离线使用吗？**
是的，完全离线运行

**Q: 内存占用？**
ASR ~500MB + TTS ~400MB + KWS ~200MB = ~1.5GB

**Q: 如何提高准确率？**
- 使用更好的麦克风
- 清理背景噪音
- 调整阈值参数

---

## 文件结构

```
VLM-Speech-Windows/
├── speech_system.py          # 核心模块
├── demo.py                   # 演示脚本
├── download_models.py        # 下载脚本
├── download_models_windows.bat
├── requirements.txt
└── README.md
```

---

## 许可证

MIT License

---

## 致谢

- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [Piper TTS](https://github.com/rhasspy/piper)
- [VLM-ROS](https://github.com/iamZhaoHang/VLM-ROS) (原始项目)

