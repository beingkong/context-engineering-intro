#!/bin/bash
set -e

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate ai-anchor

# 预下载必要模型（如果不存在）
if [ "$1" = "download-models" ]; then
    echo "📥 下载模型中..."
    python -m src.main download-models
    exit 0
fi

# 健康检查
if [ "$1" = "health-check" ]; then
    python -m src.main health-check
    exit $?
fi

# 运行测试
if [ "$1" = "test" ]; then
    echo "🧪 运行测试..."
    python -m pytest tests/ -v
    exit $?
fi

# 启动服务
if [ "$1" = "serve" ] || [ "$1" = "start" ]; then
    echo "🚀 启动AI Anchor服务..."
    
    # 检查GPU可用性
    nvidia-smi || echo "⚠️  GPU不可用，某些功能可能受限"
    
    # 检查环境变量
    if [ -z "$HOST" ]; then
        export HOST="0.0.0.0"
    fi
    if [ -z "$PORT" ]; then
        export PORT="8000"
    fi
    
    # 启动服务
    if [ "$DEBUG" = "true" ]; then
        echo "🐛 调试模式启动..."
        python -m src.main start --dev
    else
        echo "🏭 生产模式启动..."
        gunicorn -k uvicorn.workers.UvicornWorker \
            --bind ${HOST}:${PORT} \
            --workers 1 \
            --timeout 300 \
            --max-requests 1000 \
            --max-requests-jitter 50 \
            --access-logfile - \
            --error-logfile - \
            src.main:app
    fi
    
    exit $?
fi

# 默认执行参数
exec "$@"