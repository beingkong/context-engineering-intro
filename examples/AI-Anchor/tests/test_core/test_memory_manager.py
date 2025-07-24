"""
Unit tests for Memory Manager.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.memory_manager import (
    MemoryManager, 
    ModelType, 
    ModelState, 
    MemoryInfo, 
    ModelInfo,
    test_model_loading
)
from config import Settings


class TestMemoryInfo:
    """Test suite for MemoryInfo dataclass."""
    
    def test_memory_info_creation(self):
        """Test MemoryInfo creation and conversion."""
        memory_info = MemoryInfo(
            allocated_gb=5.2,
            reserved_gb=6.1,
            max_allocated_gb=7.3,
            free_gb=41.9,
            total_gb=48.0,
            utilization_percent=12.7
        )
        
        assert memory_info.allocated_gb == 5.2
        assert memory_info.reserved_gb == 6.1
        assert memory_info.max_allocated_gb == 7.3
        assert memory_info.free_gb == 41.9
        assert memory_info.total_gb == 48.0
        assert memory_info.utilization_percent == 12.7
        
        # Test conversion to dict
        memory_dict = memory_info.to_dict()
        assert isinstance(memory_dict, dict)
        assert memory_dict["allocated_gb"] == 5.2
        assert memory_dict["total_gb"] == 48.0


class TestModelInfo:
    """Test suite for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation and conversion."""
        mock_instance = Mock()
        
        model_info = ModelInfo(
            model_type=ModelType.STT,
            state=ModelState.LOADED,
            memory_usage_gb=9.5,
            last_used=time.time(),
            load_time=time.time(),
            instance=mock_instance
        )
        
        assert model_info.model_type == ModelType.STT
        assert model_info.state == ModelState.LOADED
        assert model_info.memory_usage_gb == 9.5
        assert model_info.instance == mock_instance
        
        # Test conversion to dict
        model_dict = model_info.to_dict()
        assert isinstance(model_dict, dict)
        assert model_dict["model_type"] == "stt"
        assert model_dict["state"] == "loaded"
        assert model_dict["has_instance"] is True


class TestMemoryManager:
    """Test suite for MemoryManager class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            gpu__max_gpu_memory_gb=48.0,
            gpu__memory_fraction=0.95,
            gpu__enable_gpu_monitoring=False,
            performance__memory_check_interval=1,
        )
    
    @pytest.fixture
    def memory_manager(self, test_settings):
        """Create memory manager instance."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_device = Mock()
                    mock_device.name = "RTX 6000 Ada Generation"
                    mock_device.total_memory = 48 * 1024**3  # 48GB
                    mock_props.return_value = mock_device
                    
                    manager = MemoryManager(test_settings)
                    yield manager
                    manager.shutdown()
    
    def test_initialization(self, test_settings):
        """Test memory manager initialization."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_device = Mock()
                    mock_device.name = "Test GPU"
                    mock_device.total_memory = 48 * 1024**3
                    mock_props.return_value = mock_device
                    
                    manager = MemoryManager(test_settings)
                    
                    assert manager.settings == test_settings
                    assert manager.max_gpu_memory_gb == 48.0
                    assert manager.cuda_available is True
                    assert manager.device_count == 1
                    assert len(manager.models) == 0
                    assert manager.monitoring_active is False
                    
                    manager.shutdown()
    
    def test_initialization_no_cuda(self, test_settings):
        """Test initialization without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            manager = MemoryManager(test_settings)
            
            assert manager.cuda_available is False
            assert manager.device_count == 0
            assert len(manager.device_properties) == 0
            
            manager.shutdown()
    
    def test_get_memory_info(self, memory_manager):
        """Test getting memory information."""
        with patch('torch.cuda.memory_allocated', return_value=5.2e9):
            with patch('torch.cuda.memory_reserved', return_value=6.1e9):
                with patch('torch.cuda.max_memory_allocated', return_value=7.3e9):
                    
                    memory_info = memory_manager.get_memory_info()
                    
                    assert isinstance(memory_info, MemoryInfo)
                    assert memory_info.allocated_gb == pytest.approx(5.2, rel=1e-2)
                    assert memory_info.reserved_gb == pytest.approx(6.1, rel=1e-2)
                    assert memory_info.max_allocated_gb == pytest.approx(7.3, rel=1e-2)
                    assert memory_info.total_gb == pytest.approx(48.0, rel=1e-2)
    
    def test_get_memory_info_no_cuda(self, test_settings):
        """Test getting memory info without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            manager = MemoryManager(test_settings)
            
            memory_info = manager.get_memory_info()
            
            assert memory_info.allocated_gb == 0
            assert memory_info.reserved_gb == 0
            assert memory_info.total_gb == 0
            
            manager.shutdown()
    
    def test_get_system_memory_info(self, memory_manager):
        """Test getting system memory information."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_mem = Mock()
            mock_mem.total = 32 * 1024**3  # 32GB
            mock_mem.available = 16 * 1024**3  # 16GB
            mock_mem.used = 16 * 1024**3  # 16GB
            mock_mem.percent = 50.0
            mock_memory.return_value = mock_mem
            
            system_info = memory_manager.get_system_memory_info()
            
            assert system_info["total_gb"] == pytest.approx(32.0, rel=1e-2)
            assert system_info["available_gb"] == pytest.approx(16.0, rel=1e-2)
            assert system_info["used_gb"] == pytest.approx(16.0, rel=1e-2)
            assert system_info["percent"] == 50.0
    
    def test_estimate_memory_usage(self, memory_manager):
        """Test memory usage estimation."""
        # Test known model types
        assert memory_manager.estimate_memory_usage(ModelType.STT) == 9.5
        assert memory_manager.estimate_memory_usage(ModelType.TTS) == 24.0
        assert memory_manager.estimate_memory_usage(ModelType.LLM) == 14.0
        assert memory_manager.estimate_memory_usage(ModelType.VAD) == 0.5
    
    def test_can_load_model(self, memory_manager):
        """Test model loading feasibility check."""
        # Mock memory info with plenty of free space
        mock_memory = MemoryInfo(
            allocated_gb=5.0,
            reserved_gb=6.0,
            max_allocated_gb=7.0,
            free_gb=40.0,
            total_gb=48.0,
            utilization_percent=12.5
        )
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            # Should be able to load all models with 40GB free
            assert memory_manager.can_load_model(ModelType.STT) is True
            assert memory_manager.can_load_model(ModelType.TTS) is True
            assert memory_manager.can_load_model(ModelType.LLM) is True
            assert memory_manager.can_load_model(ModelType.VAD) is True
    
    def test_can_load_model_insufficient_memory(self, memory_manager):
        """Test model loading check with insufficient memory."""
        # Mock memory info with limited free space
        mock_memory = MemoryInfo(
            allocated_gb=40.0,
            reserved_gb=42.0,
            max_allocated_gb=45.0,
            free_gb=3.0,  # Only 3GB free
            total_gb=48.0,
            utilization_percent=87.5
        )
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            # Should not be able to load large models
            assert memory_manager.can_load_model(ModelType.STT) is False  # Needs 9.5GB
            assert memory_manager.can_load_model(ModelType.TTS) is False  # Needs 24GB
            assert memory_manager.can_load_model(ModelType.LLM) is False  # Needs 14GB
            assert memory_manager.can_load_model(ModelType.VAD) is True   # Only needs 0.5GB
    
    def test_register_model(self, memory_manager):
        """Test model registration."""
        mock_instance = Mock()
        
        # Register a model
        memory_manager.register_model(ModelType.STT, mock_instance, 9.5)
        
        # Check registration
        assert ModelType.STT in memory_manager.models
        model_info = memory_manager.models[ModelType.STT]
        assert model_info.model_type == ModelType.STT
        assert model_info.state == ModelState.LOADED
        assert model_info.memory_usage_gb == 9.5
        assert model_info.instance == mock_instance
    
    def test_register_model_estimated_memory(self, memory_manager):
        """Test model registration with estimated memory."""
        mock_instance = Mock()
        
        # Register without specifying memory usage
        memory_manager.register_model(ModelType.TTS, mock_instance)
        
        # Should use estimated memory
        model_info = memory_manager.models[ModelType.TTS]
        assert model_info.memory_usage_gb == 24.0  # TTS target
    
    def test_unregister_model(self, memory_manager):
        """Test model unregistration."""
        mock_instance = Mock()
        callback = Mock()
        memory_manager.on_model_unloaded = callback
        
        # Register then unregister
        memory_manager.register_model(ModelType.VAD, mock_instance, 0.5)
        assert ModelType.VAD in memory_manager.models
        
        memory_manager.unregister_model(ModelType.VAD)
        assert ModelType.VAD not in memory_manager.models
        callback.assert_called_once_with(ModelType.VAD)
    
    def test_update_model_usage(self, memory_manager):
        """Test updating model usage timestamp."""
        mock_instance = Mock()
        memory_manager.register_model(ModelType.LLM, mock_instance)
        
        # Get initial timestamp
        initial_time = memory_manager.models[ModelType.LLM].last_used
        
        # Wait a bit and update
        time.sleep(0.01)
        memory_manager.update_model_usage(ModelType.LLM)
        
        # Check timestamp was updated
        updated_time = memory_manager.models[ModelType.LLM].last_used
        assert updated_time > initial_time
    
    def test_get_model_info(self, memory_manager):
        """Test getting model information."""
        mock_instance = Mock()
        memory_manager.register_model(ModelType.STT, mock_instance, 9.5)
        
        # Get existing model info
        model_info = memory_manager.get_model_info(ModelType.STT)
        assert model_info is not None
        assert model_info.model_type == ModelType.STT
        
        # Get non-existent model info
        model_info = memory_manager.get_model_info(ModelType.TTS)
        assert model_info is None
    
    def test_get_all_models_info(self, memory_manager):
        """Test getting all models information."""
        # Register multiple models
        memory_manager.register_model(ModelType.STT, Mock(), 9.5)
        memory_manager.register_model(ModelType.VAD, Mock(), 0.5)
        
        all_models = memory_manager.get_all_models_info()
        
        assert len(all_models) == 2
        assert ModelType.STT in all_models
        assert ModelType.VAD in all_models
        assert ModelType.TTS not in all_models
    
    def test_memory_context(self, memory_manager):
        """Test memory context manager."""
        with patch.object(memory_manager, 'get_memory_info') as mock_get_memory:
            # Setup different memory readings
            start_memory = MemoryInfo(5.0, 6.0, 7.0, 40.0, 48.0, 12.5)
            end_memory = MemoryInfo(6.5, 7.5, 8.0, 38.5, 48.0, 15.6)
            mock_get_memory.side_effect = [start_memory, end_memory]
            
            # Use context manager
            with memory_manager.memory_context(ModelType.STT):
                # Some operation that uses memory
                pass
            
            # Should have called get_memory_info twice
            assert mock_get_memory.call_count == 2
    
    def test_cleanup_memory(self, memory_manager):
        """Test memory cleanup operations."""
        with patch('gc.collect', return_value=42) as mock_gc:
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                with patch.object(memory_manager, 'get_memory_info') as mock_get_memory:
                    # Setup memory readings showing freed memory
                    start_memory = MemoryInfo(10.0, 12.0, 15.0, 36.0, 48.0, 25.0)
                    end_memory = MemoryInfo(8.0, 10.0, 15.0, 38.0, 48.0, 20.8)
                    mock_get_memory.side_effect = [start_memory, end_memory]
                    
                    freed = memory_manager.cleanup_memory()
                    
                    assert freed == 2.0  # 12.0 - 10.0
                    mock_gc.assert_called_once()
                    mock_empty_cache.assert_called_once()
    
    def test_cleanup_memory_force(self, memory_manager):
        """Test forced memory cleanup."""
        with patch('gc.collect', return_value=42):
            with patch('torch.cuda.empty_cache'):
                with patch('torch.cuda.reset_peak_memory_stats') as mock_reset:
                    with patch.object(memory_manager, 'get_memory_info') as mock_get_memory:
                        start_memory = MemoryInfo(10.0, 12.0, 15.0, 36.0, 48.0, 25.0)
                        end_memory = MemoryInfo(8.0, 10.0, 15.0, 38.0, 48.0, 20.8)
                        mock_get_memory.side_effect = [start_memory, end_memory]
                        
                        memory_manager.cleanup_memory(force=True)
                        
                        mock_reset.assert_called_once()
    
    def test_detect_memory_leaks(self, memory_manager):
        """Test memory leak detection."""
        # Register models with known memory usage
        memory_manager.register_model(ModelType.STT, Mock(), 9.5)
        memory_manager.register_model(ModelType.VAD, Mock(), 0.5)
        
        # Mock memory info showing more usage than expected
        mock_memory = MemoryInfo(
            allocated_gb=12.0,
            reserved_gb=15.0,  # Expected: 10.0 (9.5 + 0.5), Actual: 15.0
            max_allocated_gb=18.0,
            free_gb=33.0,
            total_gb=48.0,
            utilization_percent=31.25
        )
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            leak_result = memory_manager.detect_memory_leaks()
            
            assert leak_result["status"] == "checked"
            assert leak_result["expected_memory_gb"] == 10.0  # 9.5 + 0.5
            assert leak_result["actual_memory_gb"] == 15.0
            assert leak_result["discrepancy_gb"] == 5.0
            assert leak_result["potential_leak"] is True  # 5.0 > 1.0 threshold
    
    def test_start_stop_monitoring(self, memory_manager):
        """Test starting and stopping memory monitoring."""
        # Enable monitoring
        memory_manager.enable_monitoring = True
        
        # Start monitoring
        memory_manager.start_monitoring()
        assert memory_manager.monitoring_active is True
        assert memory_manager.monitoring_thread is not None
        assert memory_manager.monitoring_thread.is_alive()
        
        # Stop monitoring
        memory_manager.stop_monitoring()
        assert memory_manager.monitoring_active is False
    
    def test_memory_statistics(self, memory_manager):
        """Test getting memory statistics."""
        # Register some models
        memory_manager.register_model(ModelType.STT, Mock(), 9.5)
        memory_manager.register_model(ModelType.LLM, Mock(), 14.0)
        
        # Mock memory info
        mock_memory = MemoryInfo(10.0, 12.0, 15.0, 36.0, 48.0, 25.0)
        mock_system = {"total_gb": 32.0, "available_gb": 16.0, "used_gb": 16.0, "percent": 50.0}
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            with patch.object(memory_manager, 'get_system_memory_info', return_value=mock_system):
                
                stats = memory_manager.get_memory_statistics()
                
                assert "current_gpu_memory" in stats
                assert "system_memory" in stats
                assert "registered_models" in stats
                assert "memory_targets" in stats
                assert stats["cuda_available"] is True
                assert stats["device_count"] == 1
                assert len(stats["registered_models"]) == 2
                assert "stt" in stats["registered_models"]
                assert "llm" in stats["registered_models"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, memory_manager):
        """Test memory manager health check."""
        # Register a model
        memory_manager.register_model(ModelType.VAD, Mock(), 0.5)
        
        # Mock memory info
        mock_memory = MemoryInfo(5.0, 6.0, 7.0, 42.0, 48.0, 12.5)
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            with patch.object(memory_manager, 'detect_memory_leaks') as mock_leak_check:
                mock_leak_check.return_value = {"status": "checked", "potential_leak": False}
                
                health = await memory_manager.health_check()
                
                assert health["cuda_available"] is True
                assert health["registered_models"] == 1
                assert health["health_status"] == "healthy"  # 12.5% utilization
                assert "memory_health" in health
                assert "leak_detection" in health
    
    @pytest.mark.asyncio
    async def test_health_check_warning(self, memory_manager):
        """Test health check with warning level memory usage."""
        # Mock high memory usage
        mock_memory = MemoryInfo(35.0, 40.0, 42.0, 8.0, 48.0, 83.3)  # >80%
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            with patch.object(memory_manager, 'detect_memory_leaks') as mock_leak_check:
                mock_leak_check.return_value = {"status": "checked", "potential_leak": False}
                
                health = await memory_manager.health_check()
                
                assert health["health_status"] == "warning"
    
    @pytest.mark.asyncio
    async def test_health_check_critical(self, memory_manager):
        """Test health check with critical memory usage."""
        # Mock critical memory usage
        mock_memory = MemoryInfo(44.0, 46.0, 47.0, 2.0, 48.0, 95.8)  # >95%
        
        with patch.object(memory_manager, 'get_memory_info', return_value=mock_memory):
            with patch.object(memory_manager, 'detect_memory_leaks') as mock_leak_check:
                mock_leak_check.return_value = {"status": "checked", "potential_leak": False}
                
                health = await memory_manager.health_check()
                
                assert health["health_status"] == "critical"


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_test_model_loading_function(self):
        """Test the test_model_loading utility function."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_device = Mock()
                    mock_device.name = "Test GPU"
                    mock_device.total_memory = 48 * 1024**3
                    mock_props.return_value = mock_device
                    
                    # Mock CUDA memory functions
                    with patch('torch.cuda.memory_allocated', return_value=1e9):
                        with patch('torch.cuda.memory_reserved', return_value=2e9):
                            with patch('torch.cuda.max_memory_allocated', return_value=3e9):
                                
                                # Should run without throwing exceptions
                                await test_model_loading()
    
    def test_memory_manager_threading(self):
        """Test memory manager thread safety."""
        settings = Settings(
            gpu__max_gpu_memory_gb=48.0,
            gpu__enable_gpu_monitoring=False,
        )
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_device = Mock()
                    mock_device.name = "Test GPU"
                    mock_device.total_memory = 48 * 1024**3
                    mock_props.return_value = mock_device
                    
                    manager = MemoryManager(settings)
                    
                    # Test concurrent model registration
                    def register_models():
                        for i in range(10):
                            manager.register_model(
                                list(ModelType)[i % len(ModelType)], 
                                Mock(), 
                                1.0
                            )
                            time.sleep(0.001)
                    
                    # Run multiple threads
                    threads = []
                    for _ in range(3):
                        thread = threading.Thread(target=register_models)
                        threads.append(thread)
                        thread.start()
                    
                    # Wait for all threads
                    for thread in threads:
                        thread.join()
                    
                    # Should have registered models successfully
                    assert len(manager.models) > 0
                    
                    manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])