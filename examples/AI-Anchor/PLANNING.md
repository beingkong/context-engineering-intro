# AI Anchor Project - Architecture & Planning

## Project Overview

**AI Anchor** is a real-time voice broadcasting system that integrates cutting-edge AI models for natural, multilingual anchoring experiences. Built for RTX6000 48GB GPU optimization with <2 second end-to-end latency.

## Architecture Philosophy

### Core Principles
- **Memory-First Design**: GPU memory optimization for concurrent model operation
- **Real-time Performance**: Sub-2-second voice-to-voice pipeline
- **Modular Integration**: Clean separation of STT, TTS, LLM, and VAD components
- **Scalable Threading**: Async/threaded architecture for audio streaming
- **Environment Isolation**: Strict conda+uv dependency management

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │◄──►│  FastAPI Server │◄──►│ Audio Pipeline  │
│   (Browser)     │    │   WebSocket     │    │   Manager       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                        ┌──────────────────────────────┼──────────────────────────────┐
                        │                             │                              │
                ┌───────▼────────┐          ┌─────────▼────────┐          ┌─────────▼────────┐
                │  STT Module    │          │   LLM Module     │          │  TTS Module      │
                │  Voxtral-Mini  │          │ Ollama/huihui-ai │          │  higgs-audio v2  │
                │   (~9.5GB)     │          │   (~14-16GB)     │          │    (~24GB)       │
                └────────────────┘          └──────────────────┘          └──────────────────┘
                        │                             │                              │
                        └──────────────────────────────┼──────────────────────────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  VAD Module    │
                                              │  (TurnDetect)  │
                                              │   (<0.5GB)     │
                                              └────────────────┘
```

## Technology Stack

### AI Models
- **STT**: mistralai/Voxtral-Mini-3B-2507 (via vLLM)
  - 8 languages, 32k context, Q&A capabilities
  - Memory: ~9.5GB GPU, Temperature: 0.0 for transcription
- **TTS**: higgs-audio v2 (bosonai/higgs-audio-v2-generation-3B-base)
  - Zero-shot voice cloning, multi-speaker dialog
  - Memory: ~24GB GPU, Custom audio tokenizer
- **LLM**: huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M
  - Via Ollama, 24B parameters, uncensored responses  
  - Memory: ~14-16GB, Prompt format: `<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{prompt}[/INST]`
- **VAD**: TurnDetection (adapted from RealtimeVoiceChat)
  - DistilBert-based sentence completion, dynamic pause calculation

### Infrastructure
- **Environment**: conda + uv package management
- **Framework**: FastAPI + WebSocket for real-time communication
- **GPU**: NVIDIA RTX6000 48GB (CUDA 12.1)
- **Audio**: 24kHz sample rate, 16-bit PCM
- **Deployment**: Docker containerization support

## Module Design

### 1. Models Layer (`src/models/`)

#### STT Module (`stt_module.py`)
```python
class VoxtralSTT:
    """Voxtral-Mini-3B-2507 integration via vLLM"""
    - Real-time audio chunk processing
    - Multilingual transcription (8 languages)
    - Automatic language detection
    - Streaming response generation
```

#### TTS Module (`tts_module.py`)
```python
class HiggsAudioTTS:
    """higgs-audio v2 integration"""
    - Zero-shot voice cloning
    - Multi-speaker dialog generation
    - Real-time audio synthesis
    - Audio chunk streaming
```

#### LLM Module (`llm_module.py`)
```python
class OllamaLLM:
    """Ollama + huihui-ai model integration"""
    - Conversation history management
    - Streaming response generation
    - Custom prompt formatting
    - Context window management
```

#### VAD Module (`vad_module.py`)
```python
class VADModule:
    """Voice Activity Detection (adapted from turndetect.py)"""
    - Sentence completion prediction
    - Dynamic pause calculation
    - Context-aware turn detection
    - Threading-safe processing
```

### 2. Core Layer (`src/core/`)

#### Audio Pipeline (`audio_pipeline.py`)
```python
class AudioPipelineManager:
    """Main orchestration engine"""
    - Worker thread management (STT, TTS, LLM, VAD)
    - Audio chunk buffering and streaming
    - Memory management and model coordination
    - Error handling and recovery
```

#### Anchor Agent (`anchor_agent.py`)
```python
class AnchorAgent:
    """High-level conversation management"""
    - Personality and voice profile management
    - Conversation state tracking
    - Multi-language switching
    - Context-aware responses
```

#### Memory Manager (`memory_manager.py`)
```python
class MemoryManager:
    """GPU memory optimization"""
    - Model loading/unloading strategies
    - Memory usage monitoring
    - CUDA memory pool management
    - OOM prevention and recovery
```

### 3. Web Layer (`src/web/`)

#### Server (`server.py`)
```python
class FastAPIServer:
    """Web server and API endpoints"""
    - WebSocket audio streaming
    - RESTful API for configuration
    - Static file serving
    - CORS and security headers
```

#### WebSocket Handler (`websocket_handler.py`)
```python
class AudioWebSocketManager:
    """Real-time audio communication"""
    - Bidirectional audio streaming
    - Connection management
    - Protocol handling
    - Error recovery
```

## Data Flow Architecture

### Request Processing Pipeline
```
Audio Input → STT → VAD → LLM → TTS → Audio Output
     ↓         ↓     ↓     ↓     ↓         ↓
  PCM Bytes  Text  Pause  Text  Audio   PCM Bytes
             ↓     Time   ↓     Chunks     ↓
          Language      Context         WebSocket
          Detection     Analysis        Streaming
```

### Threading Model
```
Main Thread
├── FastAPI Server (uvicorn)
├── WebSocket Manager
└── Audio Pipeline Manager
    ├── STT Worker Thread
    ├── LLM Worker Thread  
    ├── TTS Worker Thread
    ├── VAD Worker Thread
    └── Audio Streaming Thread
```

## Memory Management Strategy

### GPU Memory Allocation
```
Total RTX6000: 48GB
├── STT (Voxtral): 9.5GB (20%)
├── TTS (higgs-audio): 24GB (50%)  
├── LLM (Mistral): 14-16GB (30%)
└── System Reserve: 2-4GB
```

### Loading Strategy
1. **Sequential Loading**: TTS → LLM → STT (largest first)
2. **Lazy Initialization**: Load models on first use
3. **Memory Monitoring**: Continuous GPU memory tracking
4. **Fallback Strategy**: Model swapping if OOM detected

## Performance Targets

### Latency Requirements
- **STT Processing**: <300ms per audio chunk
- **LLM Generation**: <800ms for first token
- **TTS Synthesis**: <500ms for first audio chunk
- **VAD Processing**: <50ms per text segment
- **Total End-to-End**: <2000ms

### Throughput Targets
- **Audio Processing**: Real-time (1x speed minimum)
- **Concurrent Sessions**: 1 primary (expandable)
- **Languages Supported**: 8 (EN, ES, FR, PT, HI, DE, NL, IT)
- **Voice Profiles**: Multiple via zero-shot cloning

## Error Handling Strategy

### Model Failures
- **STT Failure**: Fallback to simple VAD
- **TTS Failure**: Text-only response mode
- **LLM Failure**: Pre-recorded responses
- **Memory OOM**: Model reloading/swapping

### Network Issues
- **WebSocket Disconnect**: Automatic reconnection
- **Audio Buffer Underrun**: Silence padding
- **Chunk Loss**: Request retransmission

### Recovery Mechanisms
- **Graceful Degradation**: Maintain partial functionality
- **State Preservation**: Conversation context retention
- **Automatic Restart**: Component-level recovery
- **Health Monitoring**: Proactive issue detection

## Configuration Management

### Environment Variables
```bash
# Model Paths
VOXTRAL_MODEL_PATH=mistralai/Voxtral-Mini-3B-2507
HIGGS_AUDIO_MODEL_PATH=bosonai/higgs-audio-v2-generation-3B-base
OLLAMA_MODEL_NAME=hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M

# GPU Settings
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_LIMIT_GB=48
ENABLE_MEMORY_POOL=true

# Audio Settings
SAMPLE_RATE=24000
AUDIO_CHUNK_SIZE=4096
BUFFER_SIZE_MS=500

# Performance Settings
MAX_CONCURRENT_SESSIONS=1
STT_BATCH_SIZE=1
TTS_STREAM_CHUNK_SIZE=8
```

### Configuration Schema
```python
class AIAnchorConfig(BaseSettings):
    # Model Configuration
    voxtral_model_path: str
    higgs_model_path: str
    ollama_model_name: str
    
    # GPU Configuration
    gpu_memory_limit_gb: int = 48
    cuda_device_id: int = 0
    
    # Audio Configuration
    sample_rate: int = 24000
    audio_chunk_size: int = 4096
    
    # Performance Configuration  
    max_latency_ms: int = 2000
    enable_voice_cloning: bool = True
```

## Testing Strategy

### Unit Testing
- **Model Integration**: Each model component
- **Core Functionality**: Pipeline management, memory handling
- **Utilities**: Audio processing, text handling
- **Configuration**: Settings validation

### Integration Testing
- **End-to-End Pipeline**: Full voice-to-voice flow
- **Memory Management**: Concurrent model loading
- **WebSocket Communication**: Real-time audio streaming
- **Error Recovery**: Failure simulation and recovery

### Performance Testing
- **Latency Benchmarks**: Component and end-to-end timing
- **Memory Usage**: GPU allocation and optimization
- **Stress Testing**: Continuous operation stability
- **Load Testing**: Multiple session handling

### Validation Requirements
- **Code Coverage**: >80% test coverage
- **Performance**: <2s end-to-end latency
- **Memory**: <48GB total GPU usage
- **Quality**: Real-time audio fidelity

## Deployment Architecture

### Development Environment
```bash
conda create -n ai-anchor python=3.10
conda activate ai-anchor
conda install uv
uv pip install -r requirements.txt
```

### Production Container
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04
# conda + uv installation
# Model downloads and caching
# Application setup
```

### Kubernetes Deployment (Future)
- **GPU Node Affinity**: RTX6000 specific scheduling  
- **Model Volume Mounts**: Persistent model storage
- **Resource Limits**: GPU memory constraints
- **Health Checks**: Component monitoring

## Security Considerations

### Model Security
- **Model Integrity**: Checksum validation
- **Access Control**: API key authentication
- **Rate Limiting**: Request throttling
- **Input Validation**: Audio format verification

### Network Security  
- **WebSocket Security**: WSS encryption
- **CORS Policy**: Restricted origins
- **Content Security**: Audio content filtering
- **Logging**: Security event monitoring

## Monitoring & Observability

### Metrics Collection
- **Performance Metrics**: Latency, throughput, error rates
- **Resource Metrics**: GPU/CPU/memory usage
- **Business Metrics**: Session duration, language usage
- **Model Metrics**: Inference times, quality scores

### Logging Strategy
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: DEBUG, INFO, WARN, ERROR, CRITICAL
- **Component Tagging**: STT, TTS, LLM, VAD identification
- **Performance Logging**: Timing and resource usage

### Alerting
- **Latency Alerts**: >2s end-to-end response time
- **Memory Alerts**: >90% GPU memory usage
- **Error Alerts**: Model failures or connection issues
- **Health Checks**: Component availability monitoring

## Future Enhancements

### Model Improvements
- **Model Updates**: Newer versions of STT/TTS/LLM models
- **Quantization**: Further memory optimization
- **Multi-GPU**: Distributed model loading
- **Edge Deployment**: Optimized mobile versions

### Feature Additions
- **Multi-Session**: Concurrent conversation support
- **Voice Analytics**: Emotion and sentiment detection
- **Content Moderation**: Real-time content filtering
- **Advanced VAD**: Noise robustness improvements

### Infrastructure
- **Auto-Scaling**: Dynamic resource allocation
- **CDN Integration**: Global audio streaming
- **Database Integration**: Conversation persistence
- **API Gateway**: Advanced routing and security