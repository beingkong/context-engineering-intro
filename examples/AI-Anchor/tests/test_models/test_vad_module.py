"""
Unit tests for VAD module (TurnDetectionVAD).
"""

import pytest
import asyncio
import time
import threading
import collections
from unittest.mock import Mock, patch, MagicMock
from queue import Queue, Empty

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.vad_module import (
    TurnDetectionVAD, 
    ends_with_string, 
    preprocess_text, 
    strip_ending_punctuation,
    find_matching_texts,
    interpolate_detection,
    SENTENCE_END_MARKS,
    ANCHOR_POINTS
)
from config import Settings


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_ends_with_string(self):
        """Test ends_with_string function."""
        # Basic ending
        assert ends_with_string("Hello world.", ".") is True
        assert ends_with_string("Hello world!", "!") is True
        assert ends_with_string("Hello world?", "?") is True
        
        # With trailing space
        assert ends_with_string("Hello world. ", ".") is True
        assert ends_with_string("Hello world! ", "!") is True
        
        # Not ending with
        assert ends_with_string("Hello world", ".") is False
        assert ends_with_string("Hello world.", "!") is False
        
        # Edge cases
        assert ends_with_string("", ".") is False
        assert ends_with_string(".", ".") is True
    
    def test_preprocess_text(self):
        """Test preprocess_text function."""
        # Basic preprocessing
        assert preprocess_text("  hello world") == "Hello world"
        assert preprocess_text("...hello world") == "Hello world"
        assert preprocess_text("  ...  hello world") == "Hello world"
        
        # Empty string
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""
        
        # Only ellipses
        assert preprocess_text("...") == ""
        assert preprocess_text("  ...  ") == ""
        
        # Already properly formatted
        assert preprocess_text("Hello world") == "Hello world"
    
    def test_strip_ending_punctuation(self):
        """Test strip_ending_punctuation function."""
        # Single punctuation
        assert strip_ending_punctuation("Hello world.") == "Hello world"
        assert strip_ending_punctuation("Hello world!") == "Hello world"
        assert strip_ending_punctuation("Hello world?") == "Hello world"
        
        # Multiple punctuation
        assert strip_ending_punctuation("Hello world!!!") == "Hello world"
        assert strip_ending_punctuation("Hello world...") == "Hello world"
        
        # With trailing whitespace
        assert strip_ending_punctuation("Hello world.  ") == "Hello world"
        
        # No punctuation
        assert strip_ending_punctuation("Hello world") == "Hello world"
        
        # Empty string
        assert strip_ending_punctuation("") == ""
    
    def test_find_matching_texts(self):
        """Test find_matching_texts function."""
        # Create test deque
        texts_deque = collections.deque(maxlen=10)
        
        # Add some texts
        texts_deque.append(("Hello world.", "Hello world"))
        texts_deque.append(("Hello world!", "Hello world"))
        texts_deque.append(("Hello world?", "Hello world"))
        texts_deque.append(("Different text.", "Different text"))
        
        # Should find the last 3 matching "Hello world" entries
        matches = find_matching_texts(texts_deque)
        
        assert len(matches) == 1  # Only the last entry matches itself
        assert matches[0][0] == "Different text."
        
        # Test with all matching
        texts_deque.clear()
        texts_deque.append(("Same text.", "Same text"))
        texts_deque.append(("Same text!", "Same text"))
        texts_deque.append(("Same text?", "Same text"))
        
        matches = find_matching_texts(texts_deque)
        assert len(matches) == 3
        assert all(match[1] == "Same text" for match in matches)
        
        # Test with empty deque
        texts_deque.clear()
        matches = find_matching_texts(texts_deque)
        assert len(matches) == 0
    
    def test_interpolate_detection(self):
        """Test interpolate_detection function."""
        # Test with default anchor points: (0.0, 1.0), (1.0, 0.0)
        assert interpolate_detection(0.0) == 1.0
        assert interpolate_detection(1.0) == 0.0
        assert interpolate_detection(0.5) == 0.5
        
        # Test clamping
        assert interpolate_detection(-0.5) == 1.0  # Clamped to 0.0
        assert interpolate_detection(1.5) == 0.0   # Clamped to 1.0


class TestTurnDetectionVAD:
    """Test suite for TurnDetectionVAD class."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(
            debug=True,
            audio__vad_threshold=0.5,
            audio__min_pause_duration=0.5,
            audio__max_pause_duration=3.0,
        )
    
    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers components."""
        with patch('transformers.DistilBertTokenizerFast') as mock_tokenizer_class:
            with patch('transformers.DistilBertForSequenceClassification') as mock_model_class:
                # Setup mock tokenizer
                mock_tokenizer = Mock()
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                
                # Setup mock model
                mock_model = Mock()
                mock_model_class.from_pretrained.return_value = mock_model
                
                # Mock model output
                mock_outputs = Mock()
                mock_logits = Mock()
                mock_logits.shape = [1, 2]  # Batch size 1, 2 classes
                mock_outputs.logits = mock_logits
                mock_model.return_value = mock_outputs
                
                yield {
                    'tokenizer': mock_tokenizer,
                    'model': mock_model,
                    'outputs': mock_outputs
                }
    
    def test_initialization(self, test_settings, mock_transformers):
        """Test VAD initialization."""
        callback = Mock()
        
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(
                settings=test_settings,
                on_new_waiting_time=callback,
                local=True,
                pipeline_latency=0.5,
            )
        
        assert vad.settings == test_settings
        assert vad.on_new_waiting_time == callback
        assert vad.pipeline_latency == 0.5
        assert vad.current_waiting_time == -1
        assert isinstance(vad.text_time_deque, collections.deque)
        assert isinstance(vad.texts_without_punctuation, collections.deque)
        assert vad.text_worker.is_alive()
        assert vad.shutdown_flag is False
    
    def test_update_settings(self, test_settings, mock_transformers):
        """Test updating VAD settings."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
        
        # Test speed factor 0.0 (fast)
        vad.update_settings(0.0)
        fast_detection_speed = vad.detection_speed
        
        # Test speed factor 1.0 (slow)
        vad.update_settings(1.0)
        slow_detection_speed = vad.detection_speed
        
        # Slow should be greater than fast
        assert slow_detection_speed > fast_detection_speed
        
        # Test clamping
        vad.update_settings(-0.5)  # Should be clamped to 0.0
        assert vad.detection_speed == fast_detection_speed
        
        vad.update_settings(1.5)   # Should be clamped to 1.0
        assert vad.detection_speed == slow_detection_speed
        
        vad.shutdown()
    
    def test_suggest_time(self, test_settings, mock_transformers):
        """Test time suggestion functionality."""
        callback = Mock()
        
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(
                settings=test_settings,
                on_new_waiting_time=callback,
            )
        
        # Test initial suggestion
        vad.suggest_time(1.5, "Test text")
        assert vad.current_waiting_time == 1.5
        callback.assert_called_once_with(1.5, "Test text")
        
        # Test no change (callback shouldn't be called again)
        callback.reset_mock()
        vad.suggest_time(1.5, "Same time")
        callback.assert_not_called()
        
        # Test change
        vad.suggest_time(2.0, "New time")
        callback.assert_called_once_with(2.0, "New time")
        
        vad.shutdown()
    
    def test_get_suggested_whisper_pause(self, test_settings, mock_transformers):
        """Test punctuation-based pause suggestions."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
        
        # Test different punctuation marks
        assert vad.get_suggested_whisper_pause("Hello world.") == vad.punctuation_pause
        assert vad.get_suggested_whisper_pause("Hello world!") == vad.exclamation_pause
        assert vad.get_suggested_whisper_pause("Hello world?") == vad.question_pause
        assert vad.get_suggested_whisper_pause("Hello world...") == vad.ellipsis_pause
        assert vad.get_suggested_whisper_pause("Hello world") == vad.unknown_sentence_detection_pause
        
        vad.shutdown()
    
    def test_get_completion_probability_caching(self, test_settings, mock_transformers):
        """Test completion probability calculation with caching."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                with patch('torch.softmax') as mock_softmax:
                    # Mock softmax output
                    mock_probs = Mock()
                    mock_probs.__getitem__.return_value.__getitem__.return_value.item.return_value = 0.75
                    mock_softmax.return_value = [mock_probs]
                    
                    vad = TurnDetectionVAD(settings=test_settings)
                    
                    # First call should use model
                    prob1 = vad.get_completion_probability("Test sentence")
                    assert prob1 == 0.75
                    assert len(vad._completion_probability_cache) == 1
                    
                    # Second call should use cache
                    prob2 = vad.get_completion_probability("Test sentence")
                    assert prob2 == 0.75
                    assert len(vad._completion_probability_cache) == 1
                    
                    # Different text should use model again
                    prob3 = vad.get_completion_probability("Different sentence")
                    assert prob3 == 0.75
                    assert len(vad._completion_probability_cache) == 2
        
        vad.shutdown()
    
    def test_get_completion_probability_empty_text(self, test_settings, mock_transformers):
        """Test completion probability with empty text."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            assert vad.get_completion_probability("") == 0.0
            assert vad.get_completion_probability("   ") == 0.0
            
            vad.shutdown()
    
    def test_calculate_waiting_time(self, test_settings, mock_transformers):
        """Test queueing text for analysis."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            # Test normal queueing
            initial_size = vad.text_queue.qsize()
            vad.calculate_waiting_time("Test text")
            
            # Queue size should increase
            assert vad.text_queue.qsize() == initial_size + 1
            
            # Test empty text (should not queue)
            vad.calculate_waiting_time("")
            vad.calculate_waiting_time("   ")
            assert vad.text_queue.qsize() == initial_size + 1  # No change
            
            vad.shutdown()
    
    def test_reset(self, test_settings, mock_transformers):
        """Test VAD state reset."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            # Add some state
            vad.text_time_deque.append((time.time(), "test"))
            vad.texts_without_punctuation.append(("test", "test"))
            vad._completion_probability_cache["test"] = 0.5
            vad.current_waiting_time = 2.0
            
            # Reset
            vad.reset()
            
            # Check everything is cleared
            assert len(vad.text_time_deque) == 0
            assert len(vad.texts_without_punctuation) == 0
            assert len(vad._completion_probability_cache) == 0
            assert vad.current_waiting_time == -1
            
            vad.shutdown()
    
    def test_get_statistics(self, test_settings, mock_transformers):
        """Test getting VAD statistics."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            stats = vad.get_statistics()
            
            assert isinstance(stats, dict)
            assert "current_waiting_time" in stats
            assert "text_history_size" in stats
            assert "punctuation_history_size" in stats
            assert "cache_size" in stats
            assert "queue_size" in stats
            assert "detection_speed" in stats
            assert "device" in stats
            assert "model_loaded" in stats
            
            assert stats["current_waiting_time"] == -1
            assert stats["model_loaded"] is True
            
            vad.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_settings, mock_transformers):
        """Test VAD health check."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            # Mock completion probability calculation
            with patch.object(vad, 'get_completion_probability', return_value=0.75):
                health = await vad.health_check()
            
            assert health["model_loaded"] is True
            assert health["worker_alive"] is True
            assert health["shutdown_flag"] is False
            assert "device" in health
            assert "statistics" in health
            assert health["test_result"] == "success"
            assert health["test_probability"] == 0.75
            assert "test_latency" in health
            
            vad.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_check_no_model(self, test_settings):
        """Test health check without model loaded."""
        # Create VAD without proper model initialization
        vad = TurnDetectionVAD.__new__(TurnDetectionVAD)
        vad.settings = test_settings
        vad.shutdown_flag = False
        vad.text_worker = Mock()
        vad.text_worker.is_alive.return_value = False
        
        health = await vad.health_check()
        
        assert health["model_loaded"] is False
        assert health["test_result"] == "model_not_loaded"
    
    def test_shutdown(self, test_settings, mock_transformers):
        """Test VAD shutdown process."""
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(settings=test_settings)
            
            # Add some state
            vad.calculate_waiting_time("test text")
            vad._completion_probability_cache["test"] = 0.5
            
            # Verify initial state
            assert vad.shutdown_flag is False
            assert vad.text_worker.is_alive()
            assert not vad.text_queue.empty()
            assert len(vad._completion_probability_cache) > 0
            
            # Shutdown
            vad.shutdown()
            
            # Verify shutdown state
            assert vad.shutdown_flag is True
            assert vad.text_queue.empty()
            assert len(vad._completion_probability_cache) == 0
    
    def test_text_worker_processing(self, test_settings, mock_transformers):
        """Test text worker thread processing."""
        callback = Mock()
        
        with patch('torch.cuda.is_available', return_value=False):
            vad = TurnDetectionVAD(
                settings=test_settings,
                on_new_waiting_time=callback,
            )
            
            # Mock the completion probability calculation
            with patch.object(vad, 'get_completion_probability', return_value=0.8):
                # Queue some text
                vad.calculate_waiting_time("Hello world.")
                
                # Wait a bit for processing
                time.sleep(0.1)
                
                # Check that callback was called
                assert callback.called
                
                # Get the call arguments
                args = callback.call_args[0]
                waiting_time, text = args
                
                assert isinstance(waiting_time, float)
                assert waiting_time > 0
                assert "Hello world" in text
            
            vad.shutdown()


class TestTurnDetectionVADIntegration:
    """Integration tests for TurnDetectionVAD."""
    
    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        return Settings(debug=True)
    
    def test_full_processing_pipeline(self, test_settings):
        """Test the complete processing pipeline with mocked components."""
        callback_results = []
        
        def capture_callback(waiting_time, text):
            callback_results.append((waiting_time, text))
        
        with patch('transformers.DistilBertTokenizerFast') as mock_tokenizer_class:
            with patch('transformers.DistilBertForSequenceClassification') as mock_model_class:
                with patch('torch.cuda.is_available', return_value=False):
                    # Setup mocks
                    mock_tokenizer = Mock()
                    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
                    
                    mock_model = Mock()
                    mock_model_class.from_pretrained.return_value = mock_model
                    
                    # Mock tokenizer output
                    mock_tokenizer.return_value = {
                        'input_ids': Mock(),
                        'attention_mask': Mock()
                    }
                    
                    # Mock model output  
                    mock_outputs = Mock()
                    mock_logits = Mock()
                    mock_outputs.logits = mock_logits
                    mock_model.return_value = mock_outputs
                    
                    # Mock torch operations
                    with patch('torch.no_grad'):
                        with patch('torch.softmax') as mock_softmax:
                            # Mock probability output
                            mock_prob_tensor = Mock()
                            mock_prob_tensor.item.return_value = 0.7
                            mock_prob_list = Mock()
                            mock_prob_list.__getitem__.return_value = mock_prob_tensor
                            mock_softmax.return_value = [mock_prob_list]
                            
                            # Create VAD instance
                            vad = TurnDetectionVAD(
                                settings=test_settings,
                                on_new_waiting_time=capture_callback,
                            )
                            
                            # Process some text
                            test_texts = [
                                "Hello world.",
                                "How are you?",  
                                "This is a test sentence",
                                "Final text..."
                            ]
                            
                            for text in test_texts:
                                vad.calculate_waiting_time(text)
                            
                            # Wait for processing
                            time.sleep(0.2)
                            
                            # Verify results
                            assert len(callback_results) > 0
                            
                            for waiting_time, processed_text in callback_results:
                                assert isinstance(waiting_time, float)
                                assert waiting_time > 0
                                assert isinstance(processed_text, str)
                            
                            # Test shutdown
                            vad.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])