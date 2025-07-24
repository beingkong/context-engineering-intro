#!/usr/bin/env python3
"""
Test the TTS higgs-audio fix with proper data types.
"""

import sys
import asyncio
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_higgs_audio_fix():
    """Test higgs-audio with proper ChatMLSample format."""
    print("🗣️ Testing higgs-audio TTS fix...")
    
    try:
        # Import the correct data types
        from boson_multimodal.data_types import ChatMLSample, Message
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        print("✅ Imports successful")
        
        # Initialize engine
        engine = HiggsAudioServeEngine(
            model_name_or_path='bosonai/higgs-audio-v2-generation-3B-base',
            audio_tokenizer_name_or_path='bosonai/higgs-audio-v2-tokenizer',
            device='cuda',
            torch_dtype=torch.float16
        )
        
        print("✅ Engine initialized")
        
        # Create proper system prompt and messages
        system_prompt = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room with professional broadcasting quality.\n<|scene_desc_end|>"
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="Hello, this is a test of the higgs audio system.")
        ]
        
        sample = ChatMLSample(messages=messages)
        print("✅ ChatMLSample created successfully")
        
        # Test generation
        print("🎵 Generating audio...")
        output = engine.generate(
            chat_ml_sample=sample,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        print("✅ Generation successful!")
        print(f"Output type: {type(output)}")
        
        if hasattr(output, 'audio'):
            print(f"Audio shape: {output.audio.shape if hasattr(output.audio, 'shape') else 'N/A'}")
            print(f"Audio type: {type(output.audio)}")
        
        if hasattr(output, 'sampling_rate'):
            print(f"Sampling rate: {output.sampling_rate}")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_higgs_audio_fix()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)