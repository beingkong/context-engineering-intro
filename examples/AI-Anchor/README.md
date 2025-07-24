# AI Anchor - Real-time Voice Broadcasting System

A comprehensive AI anchor system for real-time voice broadcasting with advanced AI models, leveraging cutting-edge TTS, STT, and LLM technologies optimized for RTX6000 48GB GPU.

## Features

- **Real-time Voice-to-Voice Conversation**: <2 second end-to-end latency
- **Zero-shot Voice Cloning**: Dynamic anchor personalities using higgs-audio v2
- **Multilingual Support**: 8 languages (EN, ES, FR, PT, HI, DE, NL, IT)
- **Advanced AI Models**: 
  - STT: Voxtral-Mini-3B-2507 (via vLLM) - 8种语言支持
  - TTS: higgs-audio v2 - 零样本语音克隆
  - LLM: huihui-ai Mistral-Small-24B via Ollama - 对话生成
- **Smart Conversation Management**: Robust interruption handling and context awareness
- **GPU Optimization**: Concurrent model operation within 48GB memory limit

## 依赖说明

### 为什么使用vLLM？
**STT模块**使用Voxtral-Mini-3B-2507，这是一个多语言语音识别模型，需要vLLM框架来运行。
**LLM模块**使用Ollama作为推理后端，两者分工不同：
- vLLM: 专门用于Voxtral语音转文字
- Ollama: 专门用于文本对话生成

### 依赖文件说明
- `requirements-minimal.txt`: 核心依赖，解决版本冲突
- `requirements.txt`: 完整依赖（可能有兼容性问题）
- `requirements-core.txt`: 详细版本控制

## 快速开始

### 环境要求

- RTX6000 48GB GPU (或兼容的NVIDIA GPU)
- CUDA 12.1+
- Conda包管理器
- Python 3.10

### 安装步骤

1. **克隆仓库**:
   ```bash
   git clone <repository-url>
   cd context-engineering-intro/examples/AI-Anchor
   ```

2. **创建conda环境**:
   ```bash
   conda env create -f environment.yml
   conda activate ai-anchor
   ```

3. **安装依赖**:
   ```bash
   # 安装系统音频库
   sudo apt-get install portaudio19-dev python3-pyaudio
   
   # 安装Python依赖（解决vLLM兼容性）
   uv pip install -r requirements-minimal.txt
   ```

4. **从源码安装higgs-audio**:
   ```bash
   git clone https://github.com/boson-ai/higgs-audio.git
   cd higgs-audio
   uv pip install -e .
   cd ..
   ```

5. **配置环境变量**:
   ```bash
   cp .env.example .env
   # 编辑.env文件进行配置
   ```

6. **下载模型** (首次运行自动下载):
   ```bash
   python -m src.main download-models
   ```

### 使用方法

1. **启动服务**:
   ```bash
   # 开发模式
   python -m src.main start --dev
   
   # 生产模式
   python -m src.main start
   ```

2. **访问Web界面**:
   浏览器打开 http://localhost:8000

3. **健康检查**:
   ```bash
   python -m src.main health-check
   ```

4. **运行测试**:
   ```bash
   python -m pytest tests/ -v
   ```

### Docker部署

1. **构建镜像**:
   ```bash
   docker build -t ai-anchor .
   ```

2. **运行容器**:
   ```bash
   # 使用docker-compose
   docker-compose up -d
   
   # 或直接运行
   docker run --gpus all -p 8000:8000 --env-file .env ai-anchor
   ```

3. **查看日志**:
   ```bash
   docker-compose logs -f ai-anchor
   ```

## Architecture

```
AI-Anchor/
├── src/
│   ├── models/          # AI model integrations
│   ├── core/            # Core functionality  
│   ├── utils/           # Utility functions
│   └── web/             # Web interface
├── tests/               # Test suite
└── docs/                # Documentation
```

## Performance Targets

- **Latency**: <2000ms end-to-end response time
- **Memory**: <48GB total GPU usage
- **Quality**: High-fidelity audio synthesis
- **Reliability**: >99% uptime
- **Coverage**: >80% test coverage

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=src --cov-report=html

# Run specific test module
uv run pytest tests/test_models/ -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code  
ruff check src/ tests/ --fix

# Type checking
mypy src/
```

### Performance Benchmarking

```bash
python -m src.main benchmark
```

## Configuration

Key configuration options in `.env`:

- `MAX_GPU_MEMORY_GB`: Maximum GPU memory allocation
- `SAMPLE_RATE`: Audio sample rate (default: 24000)
- `MAX_CONVERSATION_HISTORY`: Context window size
- `SUPPORTED_LANGUAGES`: Comma-separated language codes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   - Reduce `MAX_GPU_MEMORY_GB` in `.env`
   - Enable model swapping: `ENABLE_MODEL_SWAPPING=true`

2. **Model Download Failures**:
   - Check internet connection
   - Verify Hugging Face access tokens
   - Clear cache: `rm -rf ~/.cache/huggingface`

3. **Audio Quality Issues**:
   - Verify sample rate settings
   - Check microphone permissions
   - Test with reference audio samples

### 帮助文档

- 查看 [故障排除指南](docs/troubleshooting.md)
- 查阅 [API文档](docs/API.md)
- 阅读 [架构设计](PLANNING.md)
- 在GitHub上提Issue

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- [higgs-audio](https://github.com/boson-ai/higgs-audio) for TTS capabilities
- [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) for multilingual STT
- [RealtimeVoiceChat](../RealtimeVoiceChat/) for architecture inspiration