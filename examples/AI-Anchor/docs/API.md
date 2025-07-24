# AI Anchor API 文档

## 概述

AI Anchor 提供RESTful API和WebSocket接口，用于实时语音对话和配置管理。

## 基础信息

- **Base URL**: `http://localhost:8000`
- **WebSocket URL**: `ws://localhost:8000/ws`
- **Content-Type**: `application/json`

## RESTful API

### 健康检查

#### GET /health
检查服务健康状态

**响应:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-24T07:46:50.123Z",
  "version": "1.0.0",
  "models": {
    "stt": "loaded",
    "tts": "loaded", 
    "llm": "loaded",
    "vad": "loaded"
  },
  "gpu": {
    "available": true,
    "memory_used": "24.5GB",
    "memory_total": "48.0GB"
  }
}
```

### 配置管理

#### GET /config
获取当前配置

**响应:**
```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "debug": false,
  "audio": {
    "sample_rate": 24000,
    "chunk_size": 1024,
    "format": "wav"
  },
  "conversation": {
    "max_history": 10,
    "default_language": "en",
    "supported_languages": ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]
  }
}
```

#### POST /config
更新配置

**请求体:**
```json
{
  "conversation": {
    "default_language": "zh",
    "temperature": 0.5
  }
}
```

### 对话管理

#### POST /conversation/start
开始新对话

**请求体:**
```json
{
  "personality": "professional",
  "language": "en",
  "voice_profile": "default"
}
```

**响应:**
```json
{
  "session_id": "conv_12345",
  "status": "started",
  "personality": "professional",
  "language": "en"
}
```

#### POST /conversation/end
结束对话

**请求体:**
```json
{
  "session_id": "conv_12345"
}
```

#### GET /conversation/{session_id}/history
获取对话历史

**响应:**
```json
{
  "session_id": "conv_12345",
  "history": [
    {
      "timestamp": "2025-01-24T07:46:50.123Z",
      "role": "user",
      "content": "你好"
    },
    {
      "timestamp": "2025-01-24T07:46:51.456Z", 
      "role": "assistant",
      "content": "你好！我是AI播音员，很高兴为您服务。"
    }
  ]
}
```

### 语音处理

#### POST /voice/profiles
添加语音配置文件

**请求体:**
```json
{
  "name": "my_voice",
  "reference_audio": "base64_encoded_audio_data",
  "description": "我的个人语音"
}
```

#### GET /voice/profiles
获取语音配置文件列表

**响应:**
```json
{
  "profiles": [
    {
      "name": "default",
      "description": "默认语音",
      "created_at": "2025-01-24T07:46:50.123Z"
    },
    {
      "name": "my_voice",
      "description": "我的个人语音",
      "created_at": "2025-01-24T07:47:50.123Z"
    }
  ]
}
```

### 系统统计

#### GET /stats
获取系统统计信息

**响应:**
```json
{
  "uptime": 3600,
  "requests_total": 1250,
  "active_sessions": 3,
  "models": {
    "stt": {
      "requests": 500,
      "avg_latency_ms": 280,
      "error_rate": 0.02
    },
    "tts": {
      "requests": 480,
      "avg_latency_ms": 450,
      "error_rate": 0.01
    },
    "llm": {
      "requests": 480,
      "avg_latency_ms": 750,
      "error_rate": 0.03
    }
  },
  "memory": {
    "gpu_used": "24.5GB",
    "gpu_total": "48.0GB",
    "ram_used": "8.2GB"
  }
}
```

## WebSocket API

### 连接
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### 消息格式

#### 客户端 → 服务器

**开始对话:**
```json
{
  "type": "start_conversation",
  "data": {
    "personality": "professional",
    "language": "en",
    "voice_profile": "default"
  }
}
```

**发送音频:**
```json
{
  "type": "audio_chunk",
  "data": {
    "audio": "base64_encoded_pcm_data",
    "sample_rate": 24000,
    "timestamp": 1706946410123
  }
}
```

**切换语言:**
```json
{
  "type": "switch_language", 
  "data": {
    "language": "zh"
  }
}
```

#### 服务器 → 客户端

**连接确认:**
```json
{
  "type": "connection_established",
  "data": {
    "session_id": "ws_12345",
    "supported_languages": ["en", "es", "fr", "pt", "hi", "de", "nl", "it"]
  }
}
```

**转录结果:**
```json
{
  "type": "transcription",
  "data": {
    "text": "Hello, how are you?",
    "language": "en",
    "confidence": 0.95,
    "is_partial": false
  }
}
```

**生成的响应:**
```json
{
  "type": "llm_response",
  "data": {
    "text": "I'm doing well, thank you! How can I help you today?",
    "language": "en"
  }
}
```

**合成音频:**
```json
{
  "type": "audio_chunk",
  "data": {
    "audio": "base64_encoded_pcm_data",
    "sample_rate": 24000,
    "is_final": false
  }
}
```

**错误消息:**
```json
{
  "type": "error",
  "data": {
    "code": "STT_ERROR",
    "message": "Speech recognition failed",
    "timestamp": 1706946410123
  }
}
```

## 错误代码

| 代码 | 描述 |
|------|------|
| `STT_ERROR` | 语音识别错误 |
| `TTS_ERROR` | 语音合成错误 |
| `LLM_ERROR` | 大语言模型错误 |
| `VAD_ERROR` | 语音活动检测错误 |
| `MEMORY_ERROR` | GPU内存不足 |
| `CONFIG_ERROR` | 配置错误 |
| `SESSION_ERROR` | 会话管理错误 |

## 限制

- 单次音频文件最大30秒
- 并发连接最大5个
- API请求频率限制: 100/分钟
- WebSocket消息大小限制: 16MB

## 认证

如果启用了API_KEY认证：

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
```

WebSocket连接时在URL中包含token：
```javascript
const ws = new WebSocket('ws://localhost:8000/ws?token=your-api-key');
```