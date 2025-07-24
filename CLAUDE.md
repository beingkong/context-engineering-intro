### 🔄 Project Awareness & Context
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the task isn't listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use conda environment management** for Python environment isolation (required by user specification).
- **Use uv for dependency installation** within the conda environment for faster package management.

### 🧱 Code Structure & Modularity - AI Anchor Project
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
  For AI anchor components this looks like:
    - `stt_module.py` - Voxtral-Mini-3B-2507 Speech-to-Text integration
    - `tts_module.py` - higgs-audio Text-to-Speech integration  
    - `llm_module.py` - Ollama LLM server communication
    - `audio_pipeline.py` - Audio processing and streaming coordination
    - `anchor_agent.py` - Main anchor agent logic and conversation management
    - `vad_module.py` - Voice Activity Detection (when implemented)
- **Use clear, consistent imports** (prefer relative imports within packages).
- **Use python_dotenv and load_env()** for environment variables.

### 🎤 AI Anchor Specific Architecture
- **Audio Processing Pipeline**: STT → LLM → TTS with WebSocket streaming
- **Model Integration Requirements**:
  - **STT**: Voxtral-Mini-3B-2507 via vLLM (~9.5GB GPU memory)
  - **TTS**: higgs-audio v2 (~24GB GPU memory recommended) 
  - **LLM**: Ollama with huihui-ai Mistral-Small-24B Q4_K_M (~14-16GB memory)
- **Hardware Optimization**: Leverage RTX6000 48GB for concurrent model operation
- **Real-time Performance**: Prioritize low-latency audio streaming and response generation

### 🧪 Testing & Reliability
- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use (audio processing, model integration)
    - 1 edge case (network interruption, model failure)
    - 1 failure case (invalid audio input, memory overflow)
- **Audio Testing**: Include tests for audio format validation, streaming interruption, and model response timing.

### ✅ Task Completion
- **Mark completed tasks in `TASK.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to `TASK.md` under a "Discovered During Work" section.

### 📎 Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation** especially for audio metadata and model configurations.
- Use `FastAPI` for APIs and WebSocket communication.
- Write **docstrings for every function** using the Google style:
  ```python
  def process_audio():
      """
      Brief summary of audio processing function.

      Args:
          audio_data (bytes): Raw audio input data.
          format (str): Audio format specification.

      Returns:
          ProcessedAudio: Processed audio with metadata.
      """
  ```

### 🔧 Environment & Dependencies
- **Conda Environment**: Create and manage environment with conda
- **UV Package Manager**: Use `uv pip install` for fast dependency installation
- **GPU Memory Management**: Monitor and optimize memory usage across STT/TTS/LLM models
- **Model Loading**: Implement lazy loading and memory-efficient model switching

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** especially audio processing algorithms and model integration points.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.
- **Document model configurations** including memory requirements, performance characteristics, and optimization settings.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
- **Model Integration**: Always verify model compatibility and hardware requirements before implementation.

### 🎯 AI Anchor Specific Guidelines
- **Audio Quality**: Prioritize audio fidelity and real-time performance
- **Voice Cloning**: Leverage higgs-audio's zero-shot capabilities for dynamic voice generation  
- **Multilingual Support**: Utilize Voxtral's 8-language support for international anchoring
- **Memory Efficiency**: Optimize model loading order and memory sharing between components
- **Interruption Handling**: Implement graceful interruption and conversation flow management
- **Performance Monitoring**: Track latency, memory usage, and model performance metrics