## FEATURE:

AI Anchor Project - Real-time voice interaction system with advanced AI models for broadcasting/anchoring applications.

**Core Components:**
- **Text-to-Speech**: higgs-audio v2 (zero-shot voice cloning, multi-speaker dialog)
- **Speech-to-Text**: mistralai/Voxtral-Mini-3B-2507 (multilingual, 32k context, Q&A capabilities)
- **Large Language Model**: Ollama with huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M
- **Voice Activity Detection**: To be implemented (leveraging existing turn detection patterns)
- **Hardware**: NVIDIA RTX6000 48GB GPU for concurrent model operation

## EXAMPLES:

Reference the existing `examples/RealtimeVoiceChat/` for architectural patterns and WebSocket streaming implementation. The AI anchor project will adapt this foundation with:

- **Model Integration Examples**: Located in `examples/ai_anchor/`
  - `stt_integration.py` - Voxtral-Mini-3B-2507 integration patterns
  - `tts_integration.py` - higgs-audio v2 setup and usage
  - `llm_integration.py` - Ollama communication patterns
  - `anchor_demo.py` - Complete anchor interaction demo

## DOCUMENTATION:

### Primary Model Documentation:
- **higgs-audio TTS**: https://github.com/boson-ai/higgs-audio/blob/main/README.md
- **Voxtral-Mini-3B-2507 STT**: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507  
- **huihui-ai Model**: https://huggingface.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF

### Setup Documentation:
- Conda environment management
- UV package installation
- CUDA setup for RTX6000
- Model downloading and configuration

## OTHER CONSIDERATIONS:

### Environment Management:
- **MANDATORY**: Use conda for environment isolation (user requirement)
- **MANDATORY**: Use uv for dependency installation within conda environment
- Python 3.10+ recommended for model compatibility

### Hardware Optimization:
- **GPU Memory Allocation**:
  - STT (Voxtral): ~9.5GB
  - TTS (higgs-audio): ~24GB
  - LLM (Mistral-Small): ~14-16GB  
  - Total: ~47.5GB (fits within RTX6000 48GB)
- **Concurrent Model Loading**: Implement memory-efficient model management
- **CUDA Optimization**: Ensure proper CUDA toolkit version compatibility

### Model-Specific Requirements:
- **higgs-audio**: Requires PyTorch, custom audio tokenizer, supports Docker deployment
- **Voxtral**: Requires vLLM framework, supports 8 languages, 32k context window
- **Ollama**: Local model server, requires model pull after installation
- **Memory Management**: Implement lazy loading for optimal GPU utilization

### Architecture Considerations:
- **Real-time Streaming**: Maintain low-latency audio processing pipeline
- **Interruption Handling**: Implement graceful conversation flow management  
- **Voice Cloning**: Leverage higgs-audio's zero-shot capabilities for dynamic voice generation
- **Multilingual Support**: Utilize Voxtral's 8-language support (EN, ES, FR, PT, HI, DE, NL, IT)
- **Error Recovery**: Implement robust error handling for model failures and network issues

### Installation Gotchas:
- **Conda/UV Integration**: Ensure UV works properly within conda environment
- **Model Downloads**: Large model files require stable internet and sufficient storage
- **CUDA Compatibility**: Verify PyTorch CUDA version matches system CUDA installation
- **Memory Validation**: Test concurrent model loading before production deployment
- **Audio Device Access**: Ensure proper microphone/speaker permissions for web interface