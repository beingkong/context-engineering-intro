# 故障排除指南

## 常见问题

### 1. CUDA内存不足

**症状:** 
```
RuntimeError: CUDA out of memory
```

**解决方案:**
```bash
# 1. 减少GPU内存限制
export GPU__MAX_GPU_MEMORY_GB=40

# 2. 启用模型交换
export PERFORMANCE__ENABLE_MODEL_SWAPPING=true

# 3. 检查其他GPU进程
nvidia-smi
kill -9 <进程ID>
```

### 2. 模型下载失败

**症状:**
```
ConnectionError: Failed to download model
```

**解决方案:**
```bash
# 1. 检查网络连接
ping huggingface.co

# 2. 手动下载模型
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('mistralai/Voxtral-Mini-3B-2507')
"

# 3. 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. Ollama连接失败

**症状:**
```
ConnectionError: Could not connect to Ollama
```

**解决方案:**
```bash
# 1. 启动Ollama
systemctl start ollama

# 2. 检查Ollama状态
ollama ps

# 3. 拉取模型
ollama pull hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M
```

### 4. 音频格式错误

**症状:**
```
AudioError: Unsupported audio format
```

**解决方案:**
```bash
# 使用ffmpeg转换格式
ffmpeg -i input.mp3 -ar 24000 -ac 1 -f wav output.wav

# 或在代码中设置正确格式
AUDIO__SAMPLE_RATE=24000
AUDIO__FORMAT=wav
```

### 5. WebSocket连接断开

**症状:**
```
WebSocket connection closed unexpectedly
```

**解决方案:**
```bash
# 1. 检查防火墙
sudo ufw allow 8000

# 2. 增加超时时间
PERFORMANCE__RESPONSE_TIMEOUT=60

# 3. 检查CORS设置
SECURITY__CORS_ORIGINS=["*"]
```

## 性能调优

### GPU优化
```bash
# 设置GPU性能模式
nvidia-smi -pm 1

# 设置最大功耗
nvidia-smi -pl 450

# 监控GPU使用
watch -n 1 nvidia-smi
```

### 内存优化
```bash
# 启用内存交换
export PERFORMANCE__ENABLE_MODEL_SWAPPING=true

# 调整检查间隔
export PERFORMANCE__MEMORY_CHECK_INTERVAL=10

# 清理模型缓存
python -c "
import torch
torch.cuda.empty_cache()
"
```

### 延迟优化
```bash
# 减少音频块大小
export AUDIO__CHUNK_SIZE=512

# 调整批处理大小
export STT__BATCH_SIZE=1

# 启用缓存
export PERFORMANCE__ENABLE_CACHING=true
```

## 日志分析

### 启用调试日志
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### 查看特定组件日志
```bash
# STT日志
grep "VoxtralSTT" logs/ai-anchor.log

# TTS日志  
grep "HiggsAudio" logs/ai-anchor.log

# LLM日志
grep "Ollama" logs/ai-anchor.log

# 性能日志
grep "performance" logs/ai-anchor.log
```

### 错误模式识别
```bash
# 内存错误
grep -i "memory\|oom" logs/ai-anchor.log

# 网络错误
grep -i "connection\|timeout" logs/ai-anchor.log

# 模型错误
grep -i "model\|inference" logs/ai-anchor.log
```

## 系统要求验证

### GPU检查
```bash
# CUDA版本
nvcc --version

# GPU信息
nvidia-smi --query-gpu=name,memory.total --format=csv

# CUDA库检查
python -c "import torch; print(torch.cuda.is_available())"
```

### 依赖检查
```bash
# Python版本
python --version

# 关键包版本
pip list | grep -E "(torch|transformers|vllm|soundfile)"

# 系统音频库
ldconfig -p | grep -E "(sndfile|pulse|alsa)"
```

### 端口检查
```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 测试端口连通性
telnet localhost 8000
```

## 备份与恢复

### 模型备份
```bash
# 备份模型缓存
tar -czf models-backup.tar.gz ~/.cache/huggingface ~/.ollama

# 恢复模型
tar -xzf models-backup.tar.gz -C ~
```

### 配置备份
```bash
# 备份配置
cp .env .env.backup
cp -r logs logs.backup

# 恢复配置
cp .env.backup .env
```

## 联系支持

如果问题仍未解决，请提供以下信息：

1. **系统信息:**
   ```bash
   uname -a
   nvidia-smi
   python --version
   ```

2. **错误日志:**
   ```bash
   tail -100 logs/ai-anchor.log
   ```

3. **配置文件:**
   ```bash
   cat .env (移除敏感信息)
   ```

4. **重现步骤:** 详细描述如何重现问题

请将这些信息发送至技术支持团队。