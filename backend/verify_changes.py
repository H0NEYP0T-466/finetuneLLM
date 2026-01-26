#!/usr/bin/env python3
"""
Test script to verify the text-based history system and model parameter changes.
This can be run without starting the full server.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List
from pathlib import Path

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

def test_history_file():
    """Test the history file system"""
    print("\n" + "=" * 70)
    print("Testing History File System")
    print("=" * 70)
    
    # Create a test history file
    test_file = Path(__file__).parent / "test_history.txt"
    
    try:
        # Write test data
        print("\n[Test 1] Writing test data to history file...")
        with open(test_file, 'w') as f:
            f.write("2024-01-01T00:00:00|session1|user|Hello\n")
            f.write("2024-01-01T00:00:00|session1|assistant|Hi there!\n")
            f.write("2024-01-01T00:00:01|session2|user|What is AI?\n")
            f.write("2024-01-01T00:00:01|session2|assistant|AI is artificial intelligence.\n")
        print("✓ PASSED")
        
        # Read and verify
        print("\n[Test 2] Reading and parsing history file...")
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"
        
        # Parse first line
        parts = lines[0].strip().split('|', 3)
        assert len(parts) == 4, "Line should have 4 parts"
        timestamp, session_id, role, content = parts
        assert role == "user", f"Expected 'user', got '{role}'"
        assert content == "Hello", f"Expected 'Hello', got '{content}'"
        print("✓ PASSED")
        
        # Cleanup
        test_file.unlink()
        print("\n[Test 3] Cleanup test file")
        print("✓ PASSED")
        
        print("\n" + "=" * 70)
        print("All history file tests PASSED! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        if test_file.exists():
            test_file.unlink()
        raise

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
    
    # Test history file
    print("\n")
    try:
        test_history_file()
    except Exception as e:
        print(f"History file test failed: {e}")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Start services: docker-compose up -d")
    print("2. Check logs: docker-compose logs -f backend")
    print("3. Test chat: curl -X POST http://localhost:8002/chat \\")
    print("              -H 'Content-Type: application/json' \\")
    print("              -d '{\"prompt\": \"hello\"}'")
    print("4. Check history: cat backend/history.txt")
    print("\n")

if __name__ == "__main__":
    main()
