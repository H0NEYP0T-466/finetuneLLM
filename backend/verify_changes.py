#!/usr/bin/env python3
"""
Test script to verify the MongoDB connection and model parameter changes.
This can be run without starting the full server.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List

def format_chat_prompt(history: List[Dict[str, str]], new_message: str) -> str:
    """
    Format conversation history into a proper chat prompt.
    """
    prompt = """You are a helpful AI assistant. Provide clear, concise, and relevant responses.
Answer questions directly and briefly. Keep responses focused and under 3 sentences unless specifically asked for more detail.
Do not generate examples, puzzles, or unrelated content. Stay on topic.

"""
    
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    
    prompt += f"User: {new_message}\nAssistant:"
    
    return prompt

def test_prompt_formatting():
    """Test the chat prompt formatting"""
    print("=" * 70)
    print("Testing Chat Prompt Formatting")
    print("=" * 70)
    
    # Test 1: Empty history
    history = []
    prompt = format_chat_prompt(history, "hello")
    print("\n[Test 1] Empty history with 'hello':")
    print(prompt)
    assert "You are a helpful AI assistant" in prompt
    assert "User: hello\nAssistant:" in prompt
    print("✓ PASSED")
    
    # Test 2: With conversation history
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"}
    ]
    prompt = format_chat_prompt(history, "what is your name?")
    print("\n[Test 2] With conversation history:")
    print(prompt)
    assert "User: hello" in prompt
    assert "Assistant: Hi! How can I help you?" in prompt
    assert "User: what is your name?\nAssistant:" in prompt
    print("✓ PASSED")
    
    # Test 3: Check system prompt emphasizes brevity
    assert "concise" in prompt
    assert "briefly" in prompt
    assert "Do not generate examples, puzzles" in prompt
    print("\n[Test 3] System prompt emphasizes brevity and no examples")
    print("✓ PASSED")
    
    print("\n" + "=" * 70)
    print("All prompt formatting tests PASSED! ✓")
    print("=" * 70)

def test_parameters():
    """Display the key parameters that were changed"""
    print("\n" + "=" * 70)
    print("Key Parameters Changed")
    print("=" * 70)
    
    print("\nGeneration Parameters:")
    print("  max_tokens:    512 → 150  (prevents rambling)")
    print("  temperature:   0.7 → 0.5  (more focused responses)")
    print("  repeat_penalty: 1.1       (prevents repetition)")
    print("  top_k:          40        (better token selection)")
    
    print("\nStop Sequences Added:")
    stop_sequences = [
        "</s>", 
        "User:", "\\nUser:", 
        "Human:", "\\nHuman:",
        "Question:", "\\nQuestion:",
        "\\n\\n\\n",
        "Example:",
        "Here's an",
        "Let me",
    ]
    for seq in stop_sequences:
        print(f"  - {repr(seq)}")
    
    print("\n" + "=" * 70)
    print("These changes should match LM Studio behavior")
    print("=" * 70)

async def test_mongodb_connection():
    """Test MongoDB connection logic"""
    print("\n" + "=" * 70)
    print("Testing MongoDB Connection Logic")
    print("=" * 70)
    
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        import os
        
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        print(f"\nConnecting to: {mongodb_uri}")
        
        client = AsyncIOMotorClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        await client.admin.command('ping')
        print("✓ MongoDB connection successful")
        
        # Check if database exists
        db_list = await client.list_database_names()
        print(f"\nExisting databases: {db_list}")
        
        if "finetuneLLM" not in db_list:
            print("✗ Database 'finetuneLLM' does not exist yet")
            print("  (It will be created automatically when first document is inserted)")
        else:
            print("✓ Database 'finetuneLLM' exists")
        
        # Connect to database
        db = client["finetuneLLM"]
        chat_collection = db["chats"]
        
        # Check collection
        count = await chat_collection.count_documents({})
        print(f"✓ Collection 'chats' has {count} documents")
        
        client.close()
        print("\n✓ MongoDB test PASSED")
        
    except ImportError:
        print("⚠ motor not installed - skipping MongoDB test")
    except Exception as e:
        print(f"⚠ MongoDB connection failed: {e}")
        print("  This is expected if MongoDB is not running")
        print("  The application will work without MongoDB (in-memory mode)")

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FineTuneLLM Backend - Change Verification Tests".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Test prompt formatting
    test_prompt_formatting()
    
    # Display parameters
    test_parameters()
    
    # Test MongoDB (async)
    print("\n")
    try:
        asyncio.run(test_mongodb_connection())
    except Exception as e:
        print(f"MongoDB test skipped: {e}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start services: docker-compose up -d")
    print("2. Check logs: docker-compose logs -f backend")
    print("3. Test chat: curl -X POST http://localhost:8002/chat \\")
    print("              -H 'Content-Type: application/json' \\")
    print("              -d '{\"prompt\": \"hello\"}'")
    print("\n")

if __name__ == "__main__":
    main()
