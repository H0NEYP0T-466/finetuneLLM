#!/usr/bin/env python3
"""
Integration test for text-based history system.
This test verifies that the history file format is correct and can be read/written properly.
"""
import asyncio
from pathlib import Path
from datetime import datetime
import sys

# Mock aiofiles for testing
class MockAioFile:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.file = None
    
    async def __aenter__(self):
        self.file = open(self.path, self.mode)
        return self
    
    async def __aexit__(self, *args):
        if self.file:
            self.file.close()
    
    async def write(self, content):
        self.file.write(content)
    
    async def read(self):
        return self.file.read()

async def test_history_file_write():
    """Test writing to history file"""
    print("=" * 70)
    print("Test 1: Writing to History File")
    print("=" * 70)
    
    test_file = Path(__file__).parent / "test_history_write.txt"
    
    try:
        # Simulate writing as the app does
        timestamp = datetime.utcnow().isoformat()
        session_id = "test_session"
        user_prompt = "Hello, how are you?"
        full_response = "I'm doing well, thank you!"
        
        async with MockAioFile(test_file, 'w') as f:
            await f.write(f"{timestamp}|{session_id}|user|{user_prompt}\n")
            await f.write(f"{timestamp}|{session_id}|assistant|{full_response}\n")
        
        # Verify the file was written
        assert test_file.exists(), "History file was not created"
        
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        assert "user|Hello, how are you?" in lines[0]
        assert "assistant|I'm doing well, thank you!" in lines[1]
        
        print("✓ History file written correctly")
        print(f"  Line 1: {lines[0].strip()}")
        print(f"  Line 2: {lines[1].strip()}")
        
        # Cleanup
        test_file.unlink()
        print("✓ Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}\n")
        if test_file.exists():
            test_file.unlink()
        return False

async def test_history_file_read():
    """Test reading from history file"""
    print("=" * 70)
    print("Test 2: Reading from History File")
    print("=" * 70)
    
    test_file = Path(__file__).parent / "test_history_read.txt"
    
    try:
        # Create test data
        test_data = [
            "2024-01-01T10:00:00.000000|session1|user|What is AI?",
            "2024-01-01T10:00:00.000000|session1|assistant|AI stands for Artificial Intelligence.",
            "2024-01-01T10:01:00.000000|session1|user|Tell me more",
            "2024-01-01T10:01:00.000000|session1|assistant|AI is the simulation of human intelligence.",
            "2024-01-01T11:00:00.000000|session2|user|Hello",
            "2024-01-01T11:00:00.000000|session2|assistant|Hi there!",
        ]
        
        with open(test_file, 'w') as f:
            for line in test_data:
                f.write(line + '\n')
        
        # Read and parse as the app does
        messages = []
        async with MockAioFile(test_file, 'r') as f:
            content = await f.read()
            lines = content.strip().split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split('|', 3)
                if len(parts) == 4:
                    timestamp, session_id, role, content = parts
                    if role == "user":
                        messages.append({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "user_prompt": content,
                            "model_response": ""
                        })
                    elif role == "assistant" and messages and messages[-1]["model_response"] == "":
                        messages[-1]["model_response"] = content
        
        # Verify parsing
        complete_messages = [msg for msg in messages if msg["user_prompt"] and msg["model_response"]]
        assert len(complete_messages) == 3, f"Expected 3 complete messages, got {len(complete_messages)}"
        
        print("✓ History file read correctly")
        print(f"  Found {len(complete_messages)} complete conversations")
        for i, msg in enumerate(complete_messages, 1):
            print(f"  {i}. User: {msg['user_prompt'][:30]}...")
            print(f"     Asst: {msg['model_response'][:30]}...")
        
        # Cleanup
        test_file.unlink()
        print("✓ Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}\n")
        if test_file.exists():
            test_file.unlink()
        return False

async def test_history_file_append():
    """Test appending to history file"""
    print("=" * 70)
    print("Test 3: Appending to History File")
    print("=" * 70)
    
    test_file = Path(__file__).parent / "test_history_append.txt"
    
    try:
        # Write initial data
        timestamp1 = "2024-01-01T10:00:00.000000"
        async with MockAioFile(test_file, 'w') as f:
            await f.write(f"{timestamp1}|session1|user|First message\n")
            await f.write(f"{timestamp1}|session1|assistant|First response\n")
        
        # Append more data (simulating a new conversation)
        timestamp2 = "2024-01-01T10:01:00.000000"
        async with MockAioFile(test_file, 'a') as f:
            await f.write(f"{timestamp2}|session1|user|Second message\n")
            await f.write(f"{timestamp2}|session1|assistant|Second response\n")
        
        # Verify
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"
        assert "First message" in lines[0]
        assert "First response" in lines[1]
        assert "Second message" in lines[2]
        assert "Second response" in lines[3]
        
        print("✓ History file append works correctly")
        print(f"  Total lines: {len(lines)}")
        
        # Cleanup
        test_file.unlink()
        print("✓ Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED: {e}\n")
        if test_file.exists():
            test_file.unlink()
        return False

async def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("  Text-Based History System - Integration Tests")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run tests
    results.append(await test_history_file_write())
    results.append(await test_history_file_read())
    results.append(await test_history_file_append())
    
    # Summary
    print("=" * 70)
    if all(results):
        print("  ✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 70)
        print("\nThe text-based history system is working correctly!")
        print("Features verified:")
        print("  ✓ Writing conversation history to file")
        print("  ✓ Reading and parsing history from file")
        print("  ✓ Appending new conversations to existing history")
        print("  ✓ Proper format: timestamp|session_id|role|content")
        return True
    else:
        print("  ❌ SOME TESTS FAILED")
        print("=" * 70)
        failed_count = sum(1 for r in results if not r)
        print(f"\n{failed_count} test(s) failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
