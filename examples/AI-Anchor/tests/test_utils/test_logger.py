"""
Unit tests for Logger utilities.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch, call
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.logger import (
    setup_logging,
    get_logger,
    log_performance,
    log_gpu_memory,
    LOGURU_AVAILABLE
)


class TestSetupLogging:
    """Test suite for setup_logging function."""
    
    @pytest.fixture
    def temp_log_file(self, tmp_path):
        """Create temporary log file."""
        return tmp_path / "test.log"
    
    def test_setup_logging_without_loguru(self):
        """Test setup_logging when loguru is not available."""
        with patch('utils.logger.LOGURU_AVAILABLE', False):
            with patch('logging.basicConfig') as mock_basic_config:
                setup_logging(level="INFO")
                
                mock_basic_config.assert_called_once()
                call_args = mock_basic_config.call_args[1]
                assert call_args['level'] == logging.INFO
                assert 'format' in call_args
                assert 'datefmt' in call_args
    
    def test_setup_logging_with_loguru(self):
        """Test setup_logging when loguru is available."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.logger') as mock_logger:
                setup_logging(level="DEBUG", enable_colors=True)
                
                # Check logger.remove was called
                mock_logger.remove.assert_called_once()
                
                # Check logger.add was called for console
                mock_logger.add.assert_called()
                
                # Check logger.info was called
                mock_logger.info.assert_called_with("Logging initialized at level: DEBUG")
    
    def test_setup_logging_with_file(self, temp_log_file):
        """Test setup_logging with file output."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.logger') as mock_logger:
                setup_logging(level="INFO", log_file=temp_log_file)
                
                # Should be called twice: once for console, once for file
                assert mock_logger.add.call_count >= 2
    
    def test_setup_logging_json_format(self, temp_log_file):
        """Test setup_logging with JSON format."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.logger') as mock_logger:
                setup_logging(level="INFO", log_file=temp_log_file, enable_json=True)
                
                # Check that serialize=True was used for file logging
                calls = mock_logger.add.call_args_list
                file_call = None
                for call in calls:
                    if 'serialize' in call[1]:
                        file_call = call
                        break
                
                assert file_call is not None
                assert file_call[1]['serialize'] is True
    
    def test_setup_logging_no_colors(self):
        """Test setup_logging without colors."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.logger') as mock_logger:
                setup_logging(level="INFO", enable_colors=False)
                
                # Check console handler was called
                console_call = mock_logger.add.call_args_list[0]
                assert console_call[1]['colorize'] is False


class TestGetLogger:
    """Test suite for get_logger function."""
    
    def test_get_logger_with_loguru(self):
        """Test get_logger when loguru is available."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.logger') as mock_logger:
                mock_bound_logger = Mock()
                mock_logger.bind.return_value = mock_bound_logger
                
                result = get_logger("test_module")
                
                mock_logger.bind.assert_called_once_with(name="test_module")
                assert result == mock_bound_logger
    
    def test_get_logger_without_loguru(self):
        """Test get_logger when loguru is not available."""
        with patch('utils.logger.LOGURU_AVAILABLE', False):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                result = get_logger("test_module")
                
                mock_get_logger.assert_called_once_with("test_module")
                assert result == mock_logger


class TestLogPerformance:
    """Test suite for log_performance decorator."""
    
    def test_log_performance_sync_function(self):
        """Test log_performance with synchronous function."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_performance("Test operation")
            def test_function():
                time.sleep(0.01)  # Small delay
                return "result"
            
            result = test_function()
            
            assert result == "result"
            mock_get_logger.assert_called_once_with("performance")
            mock_logger.info.assert_called_once()
            
            # Check log message contains operation name and timing
            log_call = mock_logger.info.call_args[0][0]
            assert "Test operation" in log_call
            assert "completed in" in log_call
    
    def test_log_performance_sync_function_with_error(self):
        """Test log_performance with synchronous function that raises error."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_performance("Error operation")
            def error_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError, match="Test error"):
                error_function()
            
            mock_logger.error.assert_called_once()
            
            # Check error log message
            log_call = mock_logger.error.call_args[0][0]
            assert "Error operation" in log_call
            assert "failed after" in log_call
    
    @pytest.mark.asyncio
    async def test_log_performance_async_function(self):
        """Test log_performance with asynchronous function."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_performance("Async operation")
            async def async_function():
                await asyncio.sleep(0.01)
                return "async_result"
            
            import asyncio
            result = await async_function()
            
            assert result == "async_result"
            mock_logger.info.assert_called_once()
            
            # Check log message
            log_call = mock_logger.info.call_args[0][0]
            assert "Async operation" in log_call
            assert "completed in" in log_call
    
    @pytest.mark.asyncio
    async def test_log_performance_async_function_with_error(self):
        """Test log_performance with async function that raises error."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @log_performance("Async error operation")
            async def async_error_function():
                raise RuntimeError("Async error")
            
            import asyncio
            with pytest.raises(RuntimeError, match="Async error"):
                await async_error_function()
            
            mock_logger.error.assert_called_once()
            
            # Check error log message
            log_call = mock_logger.error.call_args[0][0]
            assert "Async error operation" in log_call
            assert "failed after" in log_call
    
    def test_log_performance_with_loguru_available(self):
        """Test log_performance behavior when loguru is available."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.get_logger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                @log_performance("Loguru operation")
                def test_function():
                    return "result"
                
                result = test_function()
                
                assert result == "result"
                
                # Check emoji is included when loguru is available
                log_call = mock_logger.info.call_args[0][0]
                assert "⏱️" in log_call
    
    def test_log_performance_without_loguru(self):
        """Test log_performance behavior when loguru is not available."""
        with patch('utils.logger.LOGURU_AVAILABLE', False):
            with patch('utils.logger.get_logger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                @log_performance("Standard operation")
                def test_function():
                    return "result"
                
                result = test_function()
                
                assert result == "result"
                
                # Check no emoji when loguru is not available
                log_call = mock_logger.info.call_args[0][0]
                assert "⏱️" not in log_call
                assert "Standard operation" in log_call


class TestLogGpuMemory:
    """Test suite for log_gpu_memory function."""
    
    def test_log_gpu_memory_with_cuda(self):
        """Test log_gpu_memory when CUDA is available."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.memory_allocated', return_value=5e9):  # 5GB
                    with patch('torch.cuda.memory_reserved', return_value=6e9):  # 6GB
                        
                        log_gpu_memory("Test memory check")
                        
                        mock_get_logger.assert_called_once_with("gpu_memory")
                        mock_logger.info.assert_called_once()
                        
                        # Check log message
                        log_call = mock_logger.info.call_args[0][0]
                        assert "Test memory check" in log_call
                        assert "Allocated: 5.0GB" in log_call
                        assert "Reserved: 6.0GB" in log_call
    
    def test_log_gpu_memory_with_cuda_and_loguru(self):
        """Test log_gpu_memory with CUDA and loguru available."""
        with patch('utils.logger.LOGURU_AVAILABLE', True):
            with patch('utils.logger.get_logger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                with patch('torch.cuda.is_available', return_value=True):
                    with patch('torch.cuda.memory_allocated', return_value=3e9):
                        with patch('torch.cuda.memory_reserved', return_value=4e9):
                            
                            log_gpu_memory("GPU check with emoji")
                            
                            # Check emoji is included
                            log_call = mock_logger.info.call_args[0][0]
                            assert "🔧" in log_call
    
    def test_log_gpu_memory_without_cuda(self):
        """Test log_gpu_memory when CUDA is not available."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('torch.cuda.is_available', return_value=False):
                log_gpu_memory("No CUDA test")
                
                mock_logger.warning.assert_called_once_with("CUDA not available for memory logging")
    
    def test_log_gpu_memory_without_torch(self):
        """Test log_gpu_memory when PyTorch is not available."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Mock import error for torch
            with patch('builtins.__import__', side_effect=ImportError("No module named torch")):
                log_gpu_memory("No PyTorch test")
                
                mock_logger.warning.assert_called_once_with("PyTorch not available for GPU memory logging")
    
    def test_log_gpu_memory_default_operation_name(self):
        """Test log_gpu_memory with default operation name."""
        with patch('utils.logger.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.memory_allocated', return_value=1e9):
                    with patch('torch.cuda.memory_reserved', return_value=2e9):
                        
                        log_gpu_memory()  # No operation name provided
                        
                        # Check default operation name is used
                        log_call = mock_logger.info.call_args[0][0]
                        assert "GPU Memory" in log_call


class TestLoggerIntegration:
    """Integration tests for logger utilities."""
    
    def test_full_logging_workflow(self, tmp_path):
        """Test complete logging workflow."""
        log_file = tmp_path / "integration_test.log"
        
        # Test setup with fallback when loguru not available
        with patch('utils.logger.LOGURU_AVAILABLE', False):
            setup_logging(level="INFO", log_file=log_file)
            
            # Get logger
            logger = get_logger("integration_test")
            
            # Test basic logging
            logger.info("Integration test message")
            
            # Test performance decorator
            @log_performance("Integration operation")
            def test_operation():
                time.sleep(0.001)
                return "completed"
            
            result = test_operation()
            assert result == "completed"
    
    def test_logger_thread_safety(self):
        """Test logger thread safety."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker(worker_id):
            logger = get_logger(f"worker_{worker_id}")
            
            @log_performance(f"Worker {worker_id} operation")
            def worker_task():
                time.sleep(0.001)
                return f"worker_{worker_id}_result"
            
            result = worker_task()
            results.put(result)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check all results
        worker_results = []
        while not results.empty():
            worker_results.append(results.get())
        
        assert len(worker_results) == 5
        for i in range(5):
            assert f"worker_{i}_result" in worker_results
    
    def test_performance_decorator_preserves_function_metadata(self):
        """Test that performance decorator preserves function metadata."""
        @log_performance("Metadata test")
        def test_function_with_metadata(arg1, arg2="default"):
            """Test function docstring."""
            return arg1 + arg2
        
        # Check function metadata is preserved
        assert test_function_with_metadata.__name__ == "test_function_with_metadata"
        assert test_function_with_metadata.__doc__ == "Test function docstring."
        
        # Check function still works correctly
        result = test_function_with_metadata("hello", "_world")
        assert result == "hello_world"
    
    @pytest.mark.asyncio
    async def test_async_performance_decorator_preserves_metadata(self):
        """Test that async performance decorator preserves function metadata."""
        @log_performance("Async metadata test")
        async def async_test_function(value):
            """Async test function docstring."""
            await asyncio.sleep(0.001)
            return value * 2
        
        import asyncio
        
        # Check function metadata is preserved
        assert async_test_function.__name__ == "async_test_function"
        assert async_test_function.__doc__ == "Async test function docstring."
        
        # Check function still works correctly
        result = await async_test_function(5)
        assert result == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])