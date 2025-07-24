#!/usr/bin/env python3
"""
Test STT transcription with actual audio processing.
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_stt_transcription():
    """Test Voxtral STT with synthetic audio data."""
    print("🎤 Testing STT transcription...")
    
    try:
        from config import Settings
        from models.stt_module import VoxtralSTT
        
        print("✅ Imports successful")
        
        # Initialize settings and STT model
        settings = Settings()
        stt_model = VoxtralSTT(settings)
        
        print("✅ STT model created")
        
        # Initialize the model
        await stt_model.initialize()
        print("✅ STT model initialized")
        
        # Create synthetic speech-like audio data (white noise modulated for speech-like characteristics)
        sample_rate = 16000
        duration = 2.0  # 2 seconds for better transcription
        
        # Generate speech-like audio with formants (simplified)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create base noise
        base_signal = np.random.normal(0, 0.1, len(t))
        
        # Add speech-like formants around 800Hz, 1200Hz, 2500Hz
        formant1 = 0.3 * np.sin(2 * np.pi * 800 * t)
        formant2 = 0.2 * np.sin(2 * np.pi * 1200 * t)  
        formant3 = 0.1 * np.sin(2 * np.pi * 2500 * t)
        
        # Modulate with voice-like envelope (alternating voiced/unvoiced segments)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))  # 4Hz modulation
        
        # Combine all components
        audio_signal = (base_signal + formant1 + formant2 + formant3) * envelope
        audio_signal = np.clip(audio_signal, -1.0, 1.0) * 0.5  # Normalize and reduce volume
        
        # Convert to 16-bit PCM stereo (as expected by the system)
        audio_stereo = np.stack([audio_signal, audio_signal], axis=1)  # Duplicate for stereo
        audio_int16 = (audio_stereo * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        print(f"✅ Created synthetic speech-like audio: {len(audio_bytes)} bytes, {duration}s duration")
        
        # Test transcription
        print("🎤 Starting transcription...")
        result = await stt_model.transcribe_audio(audio_bytes)
        
        print("✅ Transcription completed!")
        print(f"Text: '{result.text}'")
        print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Duration: {result.end_time - result.start_time:.2f}s")
        
        # Test buffered processing
        print("\n🎤 Testing buffered processing...")
        
        # Feed chunks
        chunk_size = len(audio_bytes) // 4  # 4 chunks
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(audio_bytes)
            chunk = audio_bytes[start_idx:end_idx]
            
            print(f"  Feeding chunk {i+1}: {len(chunk)} bytes")
            stt_model.feed_audio_chunk(chunk)
        
        # Process buffered audio
        buffered_result = await stt_model.process_buffered_audio()
        
        if buffered_result:
            print("✅ Buffered processing completed!")
            print(f"Buffered text: '{buffered_result.text}'")
            print(f"Buffered confidence: {buffered_result.confidence:.2f}")
        else:
            print("⚠️ No result from buffered processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_stt_transcription()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)