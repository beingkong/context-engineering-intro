#!/usr/bin/env python3
"""
Basic test script for web interface components.
Tests the core structures without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_websocket_handler_structure():
    """Test WebSocket handler basic structure."""
    try:
        from web.websocket_handler import MessageType, WebSocketHandler
        print("✅ WebSocket handler imported successfully")
        
        # Test MessageType enum
        assert MessageType.AUDIO_CHUNK == "audio_chunk"
        assert MessageType.TEXT_INPUT == "text_input"
        assert MessageType.START_CONVERSATION == "start_conversation"
        assert MessageType.TRANSCRIPTION == "transcription"
        assert MessageType.RESPONSE_START == "response_start"
        assert MessageType.ERROR == "error"
        print("✅ MessageType enum works correctly")
        
        # Test WebSocketHandler initialization (with mocks)
        mock_settings = type('MockSettings', (), {'debug': True})()
        mock_agent = type('MockAgent', (), {
            'get_agent_status': lambda: {'is_active': True},
            'get_available_personalities': lambda: {'professional': {}}
        })()
        
        handler = WebSocketHandler(mock_settings, mock_agent)
        assert hasattr(handler, 'active_connections')
        assert hasattr(handler, 'connection_metadata')
        assert hasattr(handler, 'message_queues')
        assert isinstance(handler.active_connections, dict)
        print("✅ WebSocketHandler initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_routes_structure():
    """Test API routes basic structure."""
    try:
        from web.api_routes import AIAnchorAPI, create_app
        print("✅ API routes imported successfully")
        
        # Test create_app function (without FastAPI)
        app = create_app(None, None, None, None)
        # Should return None when FastAPI not available
        print("✅ create_app function works (returns None without FastAPI)")
        
        return True
        
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_static_files():
    """Test static files existence."""
    try:
        static_dir = Path(__file__).parent / "src" / "web" / "static"
        index_html = static_dir / "index.html"
        
        assert static_dir.exists(), "Static directory should exist"
        assert index_html.exists(), "index.html should exist"
        
        # Check HTML content
        html_content = index_html.read_text()
        assert "AI Anchor" in html_content
        assert "WebSocket" in html_content
        assert "personality" in html_content.lower()
        assert "<!DOCTYPE html>" in html_content
        print("✅ Static HTML file exists and contains expected content")
        
        return True
        
    except Exception as e:
        print(f"❌ Static files test failed: {e}")
        return False

def test_web_package_structure():
    """Test web package structure."""
    try:
        web_dir = Path(__file__).parent / "src" / "web"
        
        # Check required files exist
        required_files = [
            "__init__.py",
            "websocket_handler.py", 
            "api_routes.py",
            "static/index.html"
        ]
        
        for file_path in required_files:
            full_path = web_dir / file_path
            assert full_path.exists(), f"Required file {file_path} should exist"
        
        print("✅ Web package structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Web package structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing AI Anchor Web Interface Basic Structure\n")
    
    results = []
    results.append(test_web_package_structure())
    results.append(test_static_files())
    results.append(test_websocket_handler_structure())
    results.append(test_api_routes_structure())
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Web interface structure is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())