#!/usr/bin/env python3
"""
Basic test script for anchor agent structure.
Tests the core enums and dataclasses without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enums():
    """Test basic enum imports."""
    try:
        from agents.anchor_agent import AnchorPersonality, ConversationMode
        print("✅ Enums imported successfully")
        
        print(f"AnchorPersonality values: {list(AnchorPersonality)}")
        print(f"ConversationMode values: {list(ConversationMode)}")
        
        # Test enum usage
        assert AnchorPersonality.PROFESSIONAL == "professional"
        assert ConversationMode.CASUAL_CHAT == "casual_chat"
        print("✅ Enum values work correctly")
        
        return True
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        return False

def test_dataclasses():
    """Test dataclass structures."""
    try:
        from agents.anchor_agent import VoicePersonality, ConversationContext, AnchorPersonality, ConversationMode
        from enum import Enum
        
        print("✅ Dataclasses imported successfully")
        
        # Test VoicePersonality creation
        personality = VoicePersonality(
            name="Test",
            personality=AnchorPersonality.FRIENDLY,
            voice_profile="test_voice",
            language="en",  # Use string to avoid config dependency
            temperature=0.5
        )
        
        assert personality.name == "Test"
        assert personality.speaking_rate == 1.0  # Default value
        print("✅ VoicePersonality works correctly")
        
        # Test ConversationContext
        context = ConversationContext(
            session_id="test-123",
            mode=ConversationMode.INTERVIEW,
            current_personality=personality
        )
        
        assert context.session_id == "test-123"
        assert context.total_exchanges == 0
        assert len(context.conversation_history) == 0
        print("✅ ConversationContext works correctly")
        
        # Test adding exchanges
        context.add_exchange("Hello", "Hi there!")
        assert context.total_exchanges == 1
        assert len(context.conversation_history) == 1
        print("✅ Exchange tracking works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataclass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing AI Anchor Agent Basic Structure\n")
    
    results = []
    results.append(test_enums())
    results.append(test_dataclasses())
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Anchor agent structure is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())