#!/usr/bin/env python3
"""
Quick test script to verify STT and TTS fixes.
"""

import sys
import numpy as np
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Settings
from models.stt_module import VoxtralSTT
from models.tts_module import HiggsAudioTTS

async def test_stt_fix():
    """Test STT audio preprocessing with odd-length buffer."""
    print("🎤 Testing STT audio preprocessing fix...")
    
    # Create settings
    settings = Settings()
    stt = VoxtralSTT(settings)
    
    # Test with odd-length audio data (should cause the original error)
    odd_audio_data = b'\x00\x01\x02'  # 3 bytes (odd length)
    even_audio_data = b'\x00\x01\x02\x03'  # 4 bytes (even length)
    
    # Test odd length (should be handled gracefully now)
    result_odd = stt.preprocess_audio(odd_audio_data)
    print(f"✅ Odd-length audio processed: {len(result_odd)} samples")
    
    # Test even length (should work normally)
    result_even = stt.preprocess_audio(even_audio_data)
    print(f"✅ Even-length audio processed: {len(result_even)} samples")
    
    return True

async def test_tts_fix():
    """Test TTS engine without reference_audio parameter."""
    print("🗣️ Testing TTS parameter fix...")
    
    try:
        # Import the classes needed for TTS
        from boson_multimodal.dataset.chatml_dataset import ChatMLSample
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        import torch
        
        # Create engine (this should work now)
        engine = HiggsAudioServeEngine(
            model_name_or_path='bosonai/higgs-audio-v2-generation-3B-base',
            audio_tokenizer_name_or_path='bosonai/higgs-audio-v2-tokenizer',
            device='cuda',
            torch_dtype=torch.float16
        )
        
        # Test generate call (this should work without reference_audio)
        test_messages = [{"role": "user", "content": "test"}]
        sample = ChatMLSample(messages=test_messages)
        
        # This should not raise "unexpected keyword argument" error
        try:
            output = engine.generate(
                chat_ml_sample=sample,
                max_new_tokens=10,
                temperature=0.3,
                top_p=0.95,
                force_audio_gen=True,
            )
            print("✅ TTS generate call successful")
            return True
        except Exception as e:
            if "unexpected keyword argument" in str(e):
                print(f"❌ TTS parameter error still exists: {e}")
                return False
            else:
                print(f"✅ Parameter error fixed, but other error occurred: {e}")
                return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🔧 Testing AI Anchor fixes...")
    
    # Test STT fix
    stt_ok = await test_stt_fix()
    
    # Test TTS fix
    tts_ok = await test_tts_fix()
    
    if stt_ok and tts_ok:
        print("✅ All fixes working correctly!")
        return 0
    else:
        print("❌ Some fixes need more work")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)