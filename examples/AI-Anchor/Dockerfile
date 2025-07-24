# AI Anchor Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

LABEL maintainer="AI Anchor Team"
LABEL description="AI语音播音员系统 - RTX6000优化版本"

# 设置非交互模式
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# 安装conda和uv
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda install -y uv

ENV PATH=/opt/conda/bin:$PATH

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY environment.yml requirements.txt ./
COPY src/ ./src/
COPY docs/ ./docs/
COPY pytest.ini ./

# 创建conda环境并安装依赖
RUN conda env create -f environment.yml \
    && echo "conda activate ai-anchor" >> ~/.bashrc

# 激活环境并安装Python依赖
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate ai-anchor \
    && uv pip install -r requirements.txt \
    && uv pip install gunicorn[gevent]

# 安装higgs-audio源码
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate ai-anchor \
    && git clone https://github.com/boson-ai/higgs-audio.git /tmp/higgs-audio \
    && cd /tmp/higgs-audio \
    && uv pip install -e . \
    && rm -rf /tmp/higgs-audio

# 创建必要目录
RUN mkdir -p /app/logs /app/temp /root/.cache/huggingface

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动脚本
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["serve"]