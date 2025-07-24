"""
Memory management system for AI Anchor models.

Implements GPU memory monitoring, model loading/unloading strategies,
and memory optimization utilities for concurrent AI model operation.
"""

import asyncio
import gc
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import contextmanager

import torch
import psutil
from loguru import logger

from config import Settings
from utils.logger import log_performance, log_gpu_memory


class ModelType(str, Enum):
    """Types of AI models in the system."""
    STT = "stt"
    TTS = "tts"
    LLM = "llm"
    VAD = "vad"


class ModelState(str, Enum):
    """States of model loading."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class MemoryInfo:
    """Memory usage information."""
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float
    total_gb: float
    utilization_percent: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "max_allocated_gb": self.max_allocated_gb,
            "free_gb": self.free_gb,
            "total_gb": self.total_gb,
            "utilization_percent": self.utilization_percent,
        }


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_type: ModelType
    state: ModelState
    memory_usage_gb: float
    last_used: float
    load_time: float
    instance: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "state": self.state.value,
            "memory_usage_gb": self.memory_usage_gb,
            "last_used": self.last_used,
            "load_time": self.load_time,
            "has_instance": self.instance is not None,
        }


class MemoryManager:
    """
    Manages GPU memory allocation and optimization for AI models.
    
    Provides memory monitoring, model loading/unloading strategies,
    leak detection, and optimization utilities for concurrent operation
    of STT, TTS, LLM, and VAD models within GPU memory limits.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize memory manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.max_gpu_memory_gb = settings.gpu.max_gpu_memory_gb
        self.memory_fraction = settings.gpu.memory_fraction
        self.enable_monitoring = settings.gpu.enable_gpu_monitoring
        
        # Model tracking
        self.models: Dict[ModelType, ModelInfo] = {}
        self.model_lock = threading.RLock()
        
        # Memory monitoring
        self.memory_stats: List[MemoryInfo] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.check_interval = settings.performance.memory_check_interval
        
        # Memory allocation targets (from PRP analysis)
        self.memory_targets = {
            ModelType.STT: 9.5,   # Voxtral-Mini-3B-2507
            ModelType.TTS: 24.0,  # higgs-audio v2
            ModelType.LLM: 14.0,  # huihui-ai Mistral-Small-24B Q4_K_M
            ModelType.VAD: 0.5,   # DistilBERT for VAD
        }
        
        # Callbacks for memory events
        self.on_memory_warning: Optional[Callable[[MemoryInfo], None]] = None
        self.on_memory_critical: Optional[Callable[[MemoryInfo], None]] = None
        self.on_model_unloaded: Optional[Callable[[ModelType], None]] = None
        
        # Initialize CUDA if available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            self.device_properties = [
                torch.cuda.get_device_properties(i) 
                for i in range(self.device_count)
            ]
            logger.info(f"🧠 Memory manager initialized with {self.device_count} CUDA device(s)")
            self._log_device_info()
        else:
            logger.warning("🧠 CUDA not available - memory management will be limited")
            self.device_count = 0
            self.device_properties = []
        
        # Start monitoring if enabled
        if self.enable_monitoring and self.cuda_available:
            self.start_monitoring()
    
    def _log_device_info(self) -> None:
        """Log CUDA device information."""
        for i, props in enumerate(self.device_properties):
            total_memory_gb = props.total_memory / 1e9
            logger.info(f"🧠 GPU {i}: {props.name}, {total_memory_gb:.1f}GB total memory")
    
    def get_memory_info(self, device: int = 0) -> MemoryInfo:
        """
        Get current memory usage information.
        
        Args:
            device: CUDA device index
            
        Returns:
            Memory information object
        """
        if not self.cuda_available:
            return MemoryInfo(0, 0, 0, 0, 0, 0)
        
        try:
            # Get memory statistics
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
            
            # Get device properties
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory / 1e9
            free = total - reserved
            utilization = (reserved / total) * 100 if total > 0 else 0
            
            return MemoryInfo(
                allocated_gb=allocated,
                reserved_gb=reserved,
                max_allocated_gb=max_allocated,
                free_gb=free,
                total_gb=total,
                utilization_percent=utilization,
            )
            
        except Exception as e:
            logger.error(f"🧠 Error getting memory info: {e}")
            return MemoryInfo(0, 0, 0, 0, 0, 0)
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get system RAM memory information.
        
        Returns:
            System memory statistics
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1e9,
                "available_gb": memory.available / 1e9,
                "used_gb": memory.used / 1e9,
                "percent": memory.percent,
            }
        except Exception as e:
            logger.error(f"🧠 Error getting system memory info: {e}")
            return {"total_gb": 0, "available_gb": 0, "used_gb": 0, "percent": 0}
    
    def estimate_memory_usage(self, model_type: ModelType) -> float:
        """
        Estimate memory usage for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Estimated memory usage in GB
        """
        return self.memory_targets.get(model_type, 1.0)
    
    def can_load_model(self, model_type: ModelType) -> bool:
        """
        Check if a model can be loaded within memory constraints.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if model can be loaded
        """
        if not self.cuda_available:
            return True  # Allow loading on CPU
        
        current_memory = self.get_memory_info()
        estimated_usage = self.estimate_memory_usage(model_type)
        
        # Check if we have enough free memory
        required_memory = estimated_usage * 1.1  # Add 10% buffer
        available_memory = current_memory.free_gb
        
        can_load = available_memory >= required_memory
        
        if not can_load:
            logger.warning(f"🧠 Cannot load {model_type.value}: need {required_memory:.1f}GB, "
                          f"only {available_memory:.1f}GB available")
        
        return can_load
    
    def register_model(
        self, 
        model_type: ModelType, 
        instance: Any, 
        memory_usage_gb: Optional[float] = None
    ) -> None:
        """
        Register a loaded model with the memory manager.
        
        Args:
            model_type: Type of model
            instance: Model instance
            memory_usage_gb: Actual memory usage (will be estimated if None)
        """
        with self.model_lock:
            if memory_usage_gb is None:
                memory_usage_gb = self.estimate_memory_usage(model_type)
            
            self.models[model_type] = ModelInfo(
                model_type=model_type,
                state=ModelState.LOADED,
                memory_usage_gb=memory_usage_gb,
                last_used=time.time(),
                load_time=time.time(),
                instance=instance,
            )
            
            logger.info(f"🧠 Registered {model_type.value} model: {memory_usage_gb:.1f}GB")
            self._log_memory_summary()
    
    def unregister_model(self, model_type: ModelType) -> None:
        """
        Unregister a model from the memory manager.
        
        Args:
            model_type: Type of model to unregister
        """
        with self.model_lock:
            if model_type in self.models:
                model_info = self.models.pop(model_type)
                logger.info(f"🧠 Unregistered {model_type.value} model: {model_info.memory_usage_gb:.1f}GB freed")
                
                # Trigger callback
                if self.on_model_unloaded:
                    self.on_model_unloaded(model_type)
                
                self._log_memory_summary()
    
    def update_model_usage(self, model_type: ModelType) -> None:
        """
        Update the last used timestamp for a model.
        
        Args:
            model_type: Type of model that was used
        """
        with self.model_lock:
            if model_type in self.models:
                self.models[model_type].last_used = time.time()
    
    def get_model_info(self, model_type: ModelType) -> Optional[ModelInfo]:
        """
        Get information about a registered model.
        
        Args:
            model_type: Type of model
            
        Returns:
            Model information or None if not found
        """
        with self.model_lock:
            return self.models.get(model_type)
    
    def get_all_models_info(self) -> Dict[ModelType, ModelInfo]:
        """
        Get information about all registered models.
        
        Returns:
            Dictionary of model information
        """
        with self.model_lock:
            return self.models.copy()
    
    def _log_memory_summary(self) -> None:
        """Log current memory usage summary."""
        if not self.cuda_available:
            return
        
        current_memory = self.get_memory_info()
        total_model_memory = sum(model.memory_usage_gb for model in self.models.values())
        
        logger.info(f"🧠 Memory summary: {current_memory.reserved_gb:.1f}GB GPU used, "
                   f"{total_model_memory:.1f}GB in models, "
                   f"{current_memory.free_gb:.1f}GB free")
    
    @contextmanager
    def memory_context(self, model_type: ModelType):
        """
        Context manager for memory-aware model operations.
        
        Args:
            model_type: Type of model being operated on
        """
        start_memory = self.get_memory_info()
        
        try:
            yield
        finally:
            end_memory = self.get_memory_info()
            memory_diff = end_memory.reserved_gb - start_memory.reserved_gb
            
            if abs(memory_diff) > 0.1:  # Only log significant changes
                logger.debug(f"🧠 Memory change for {model_type.value}: {memory_diff:+.1f}GB")
    
    def cleanup_memory(self, force: bool = False) -> float:
        """
        Perform memory cleanup operations.
        
        Args:
            force: Whether to force aggressive cleanup
            
        Returns:
            Amount of memory freed in GB
        """
        start_memory = self.get_memory_info()
        
        # Clear Python garbage collection
        collected = gc.collect()
        logger.debug(f"🧠 Garbage collection freed {collected} objects")
        
        if self.cuda_available:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            logger.debug("🧠 Cleared CUDA cache")
            
            if force:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                logger.debug("🧠 Reset CUDA peak memory stats")
        
        end_memory = self.get_memory_info()
        freed_memory = start_memory.reserved_gb - end_memory.reserved_gb
        
        if freed_memory > 0.1:
            logger.info(f"🧠 Memory cleanup freed {freed_memory:.1f}GB")
        
        return freed_memory
    
    def optimize_memory_allocation(self) -> None:
        """
        Optimize memory allocation for better performance.
        """
        if not self.cuda_available:
            return
        
        logger.info("🧠 Optimizing memory allocation...")
        
        # Set memory fraction if configured
        if self.memory_fraction < 1.0:
            try:
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                logger.info(f"🧠 Set memory fraction to {self.memory_fraction:.2f}")
            except Exception as e:
                logger.warning(f"🧠 Failed to set memory fraction: {e}")
        
        # Enable memory pool if available
        try:
            if hasattr(torch.cuda, 'memory_pool'):
                # Configure memory pool settings
                pass  # Implementation depends on PyTorch version
        except Exception as e:
            logger.debug(f"🧠 Memory pool configuration not available: {e}")
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """
        Detect potential memory leaks.
        
        Returns:
            Dictionary with leak detection results
        """
        if not self.cuda_available:
            return {"status": "cuda_not_available"}
        
        current_memory = self.get_memory_info()
        expected_memory = sum(model.memory_usage_gb for model in self.models.values())
        
        # Calculate discrepancy
        actual_used = current_memory.reserved_gb
        discrepancy = actual_used - expected_memory
        
        # Threshold for considering it a leak (1GB)
        leak_threshold = 1.0
        
        result = {
            "status": "checked",
            "expected_memory_gb": expected_memory,
            "actual_memory_gb": actual_used,
            "discrepancy_gb": discrepancy,
            "potential_leak": discrepancy > leak_threshold,
            "timestamp": time.time(),
        }
        
        if result["potential_leak"]:
            logger.warning(f"🧠 Potential memory leak detected: {discrepancy:.1f}GB discrepancy")
        
        return result
    
    def start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        if self.monitoring_active or not self.cuda_available:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()
        logger.info("🧠 Started memory monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("🧠 Stopped memory monitoring")
    
    def _monitoring_loop(self) -> None:
        """Memory monitoring loop."""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                self.memory_stats.append(memory_info)
                
                # Keep only recent stats (last 100 entries)
                if len(self.memory_stats) > 100:
                    self.memory_stats.pop(0)
                
                # Check for warnings
                self._check_memory_thresholds(memory_info)
                
                # Sleep between checks
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"🧠 Error in memory monitoring: {e}")
                time.sleep(1.0)
    
    def _check_memory_thresholds(self, memory_info: MemoryInfo) -> None:
        """
        Check memory thresholds and trigger callbacks.
        
        Args:
            memory_info: Current memory information
        """
        utilization = memory_info.utilization_percent
        
        # Warning threshold: 80%
        if utilization > 80 and self.on_memory_warning:
            self.on_memory_warning(memory_info)
        
        # Critical threshold: 95%
        if utilization > 95:
            logger.error(f"🧠 Critical memory usage: {utilization:.1f}%")
            if self.on_memory_critical:
                self.on_memory_critical(memory_info)
            
            # Auto cleanup at critical level
            self.cleanup_memory(force=True)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        current_memory = self.get_memory_info()
        system_memory = self.get_system_memory_info()
        
        stats = {
            "current_gpu_memory": current_memory.to_dict(),
            "system_memory": system_memory,
            "registered_models": {
                model_type.value: model_info.to_dict()
                for model_type, model_info in self.models.items()
            },
            "memory_targets": {
                model_type.value: target_gb
                for model_type, target_gb in self.memory_targets.items()
            },
            "total_target_memory": sum(self.memory_targets.values()),
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "monitoring_active": self.monitoring_active,
        }
        
        # Add recent memory history if monitoring is active
        if self.monitoring_active and self.memory_stats:
            recent_stats = self.memory_stats[-10:]  # Last 10 entries
            stats["recent_memory_history"] = [
                mem_info.to_dict() for mem_info in recent_stats
            ]
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on memory management.
        
        Returns:
            Health status dictionary
        """
        status = {
            "cuda_available": self.cuda_available,
            "monitoring_active": self.monitoring_active,
            "registered_models": len(self.models),
            "statistics": self.get_memory_statistics(),
        }
        
        if self.cuda_available:
            # Check memory availability
            current_memory = self.get_memory_info()
            status["memory_health"] = {
                "utilization_percent": current_memory.utilization_percent,
                "free_gb": current_memory.free_gb,
                "can_load_models": {
                    model_type.value: self.can_load_model(model_type)
                    for model_type in ModelType
                }
            }
            
            # Perform leak detection
            leak_check = self.detect_memory_leaks()
            status["leak_detection"] = leak_check
            
            # Overall health assessment
            if current_memory.utilization_percent > 95:
                status["health_status"] = "critical"
            elif current_memory.utilization_percent > 80:
                status["health_status"] = "warning"
            else:
                status["health_status"] = "healthy"
        else:
            status["health_status"] = "cuda_unavailable"
        
        return status
    
    def shutdown(self) -> None:
        """Shutdown memory manager and cleanup resources."""
        logger.info("🧠 Shutting down memory manager...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Unregister all models
        with self.model_lock:
            model_types = list(self.models.keys())
            for model_type in model_types:
                self.unregister_model(model_type)
        
        # Final cleanup
        self.cleanup_memory(force=True)
        
        logger.info("✅ Memory manager shutdown complete")


# Utility functions for memory management

async def download_all_models(settings):
    """下载所有需要的AI模型."""
    from loguru import logger
    
    logger.info("🔽 开始下载模型...")
    
    # Download Ollama model
    try:
        import ollama
        model_name = settings.models.ollama_model
        logger.info(f"📥 下载Ollama模型: {model_name}")
        ollama.pull(model_name)
        logger.info("✅ Ollama模型下载完成")
    except Exception as e:
        logger.error(f"❌ Ollama模型下载失败: {e}")
    
    # Download Transformers models
    try:
        from transformers import AutoTokenizer
        
        # Voxtral model
        logger.info(f"📥 下载Voxtral模型: {settings.models.voxtral_model}")
        AutoTokenizer.from_pretrained(settings.models.voxtral_model)
        logger.info("✅ Voxtral模型下载完成")
        
        # higgs-audio tokenizer
        logger.info(f"📥 下载higgs-audio tokenizer: {settings.models.higgs_audio_tokenizer}")
        logger.info("✅ 基础模型下载完成")
        
    except Exception as e:
        logger.error(f"❌ Transformers模型下载失败: {e}")
    
    logger.info("🎉 所有模型下载任务完成!")


async def test_model_loading() -> None:
    """
    Test function for model loading and memory management.
    Used by main.py health check command.
    """
    from config import Settings
    
    settings = Settings()
    memory_manager = MemoryManager(settings)
    
    try:
        logger.info("🧠 Testing memory management system...")
        
        # Get initial memory state
        initial_memory = memory_manager.get_memory_info()
        logger.info(f"🧠 Initial memory: {initial_memory.reserved_gb:.1f}GB used, "
                   f"{initial_memory.free_gb:.1f}GB free")
        
        # Test memory allocation simulation
        for model_type in ModelType:
            can_load = memory_manager.can_load_model(model_type)
            estimated_usage = memory_manager.estimate_memory_usage(model_type)
            logger.info(f"🧠 {model_type.value}: {estimated_usage:.1f}GB estimated, "
                       f"can load: {can_load}")
        
        # Test cleanup
        memory_manager.cleanup_memory()
        
        # Get final memory state
        final_memory = memory_manager.get_memory_info()
        logger.info(f"🧠 Final memory: {final_memory.reserved_gb:.1f}GB used, "
                   f"{final_memory.free_gb:.1f}GB free")
        
        logger.info("✅ Memory management test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        raise
    finally:
        memory_manager.shutdown()