#!/usr/bin/env python3
"""
AI Anchor - Main Application Entry Point

Real-time voice broadcasting system with advanced AI models.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from loguru import logger
from rich.console import Console

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Settings
from utils.logger import setup_logging

app = typer.Typer(
    name="ai-anchor",
    help="AI Anchor - Real-time Voice Broadcasting System",
    add_completion=False,
)
console = Console()


@app.command()
def start(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    dev: bool = typer.Option(False, help="Enable development mode"),
    log_level: str = typer.Option("DEBUG", help="Log level"),
):
    """
    Start the AI Anchor server.
    """
    setup_logging(log_level)
    settings = Settings()
    
    logger.info("Starting AI Anchor server...")
    logger.info(f"Configuration: {settings.model_dump()}")
    
    # Import here to avoid circular imports
    from web.api_routes import create_app
    from agents.anchor_agent import AnchorAgent
    from core.memory_manager import MemoryManager
    from core.audio_pipeline import AudioPipelineManager
    
    # Initialize core components
    logger.info("🔧 初始化核心组件...")
    memory_manager = MemoryManager(settings)
    audio_pipeline = AudioPipelineManager(settings, memory_manager)
    anchor_agent = AnchorAgent(settings, memory_manager, audio_pipeline)
    
    # Initialize the audio pipeline
    logger.info("🎵 初始化音频管道...")
    asyncio.run(audio_pipeline.initialize())
    
    # Initialize the anchor agent
    logger.info("🎙️ 初始化主播代理...")
    asyncio.run(anchor_agent.initialize())
    
    server_app = create_app(settings, anchor_agent, memory_manager, audio_pipeline)
    
    uvicorn.run(
        server_app,
        host=host,
        port=port,
        reload=dev,
        log_level=log_level.lower(),
        access_log=dev,
    )


@app.command()
def test_models():
    """
    Test model loading and GPU memory allocation.
    """
    setup_logging("INFO")
    logger.info("Testing model loading and GPU allocation...")
    
    # Import test function
    from core.memory_manager import test_model_loading
    
    asyncio.run(test_model_loading())


@app.command()
def benchmark():
    """
    Run performance benchmarks.
    """
    setup_logging("INFO")
    logger.info("Running performance benchmarks...")
    
    # Import benchmark function
    from tests.benchmark import run_benchmarks
    
    asyncio.run(run_benchmarks())


@app.command("download-models")
def download_models():
    """
    Download and cache AI models.
    """
    setup_logging("INFO")
    logger.info("🔽 开始下载AI模型...")
    
    try:
        settings = Settings()
        
        # Import download function
        from core.memory_manager import download_all_models
        
        asyncio.run(download_all_models(settings))
        logger.info("✅ 所有模型下载完成")
        
    except Exception as e:
        logger.error(f"❌ 模型下载失败: {e}")
        raise typer.Exit(1)


@app.command("health-check")
def health_check():
    """
    Perform system health check.
    """
    setup_logging("INFO")
    
    try:
        import torch
        import vllm
        import ollama
        
        logger.info("✓ All required dependencies installed")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ CUDA available: {gpu_count} GPU(s), {gpu_memory:.1f}GB memory")
        else:
            logger.warning("⚠ CUDA not available")
            
        logger.info("✓ Health check passed")
        
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()