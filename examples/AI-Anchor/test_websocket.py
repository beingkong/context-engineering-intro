#!/usr/bin/env python3
"""
简单的WebSocket客户端用于测试文本输入到LLM的流程
"""

import asyncio
import json
import websockets
import sys

async def test_websocket():
    uri = "ws://localhost:8000/ws/test-client-123"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # 1. Start conversation
            start_msg = {
                "type": "start_conversation",
                "data": {
                    "mode": "casual_chat",
                    "personality": "professional", 
                    "language": "en"
                }
            }
            await websocket.send(json.dumps(start_msg))
            print("📤 Sent start_conversation")
            
            # Wait for response
            response = await websocket.recv()
            print("📥 Received:", json.loads(response))
            
            # 2. Send text input
            await asyncio.sleep(1)
            text_msg = {
                "type": "text_input",
                "data": {
                    "text": "Hello, how are you?"
                }
            }
            await websocket.send(json.dumps(text_msg))
            print("📤 Sent text_input: 'Hello, how are you?'")
            
            # 3. Wait for LLM response
            print("⏳ Waiting for LLM response...")
            for i in range(10):  # Wait up to 10 seconds
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message = json.loads(response)
                    print(f"📥 Received: {message.get('type')} - {message.get('data')}")
                    
                    if message.get('type') == 'llm_response':
                        print("🎉 LLM response received successfully!")
                        break
                except asyncio.TimeoutError:
                    print(f"⏳ Still waiting... ({i+1}/10)")
                    continue
            else:
                print("⚠️ No LLM response received within 10 seconds")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_websocket())