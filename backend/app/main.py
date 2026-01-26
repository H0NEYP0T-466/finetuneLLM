from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from rich.console import Console
from rich.logging import RichHandler
import logging
import time
import os
from pathlib import Path
from typing import Optional, List, Dict
import json
from datetime import datetime
from contextlib import asynccontextmanager
from collections import defaultdict
import aiofiles

# Setup Rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger(__name__)

# Global variables
llm_model: Optional[Llama] = None

# Path to history file
HISTORY_FILE = Path(__file__).parent.parent / "history.txt"

# Store conversation history per session (in-memory for simplicity)
# Format: {session_id: [{"role": "user/assistant", "content": "..."}]}
conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_model
    
    # Ensure history file exists
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not HISTORY_FILE.exists():
            HISTORY_FILE.touch()
            logger.info("[bold green]✓ Created history.txt file[/bold green]", extra={"markup": True})
        else:
            logger.info("[bold cyan]history.txt file already exists[/bold cyan]", extra={"markup": True})
    except Exception as e:
        logger.warning(f"[bold yellow]⚠ Could not create history file: {e}[/bold yellow]", extra={"markup": True})
    
    # Load LLM model
    model_path = Path(__file__).parent.parent / "model"
    gguf_files = list(model_path.glob("*.gguf"))
    
    if not gguf_files:
        logger.warning("[bold yellow]⚠ No GGUF model files found in model directory[/bold yellow]", extra={"markup": True})
        logger.warning("[bold yellow]Please place a .gguf model file in the backend/model directory[/bold yellow]", extra={"markup": True})
    else:
        model_file = gguf_files[0]
        logger.info(f"[bold cyan]Loading model: {model_file.name}[/bold cyan]", extra={"markup": True})
        
        start_time = time.time()
        try:
            llm_model = Llama(
                model_path=str(model_file),
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=-1,  # Use GPU if available
                verbose=False,
                # Prevents the model from repeating tokens in loops
                repeat_penalty=1.1,
            )
            load_time = time.time() - start_time
            logger.info(f"[bold green]✓ Model loaded successfully in {load_time:.2f} seconds[/bold green]", extra={"markup": True})
        except Exception as e:
            logger.error(f"[bold red]✗ Model loading failed: {e}[/bold red]", extra={"markup": True})
    
    yield
    
    # Shutdown
    logger.info("[bold cyan]Application shutdown[/bold cyan]", extra={"markup": True})

app = FastAPI(title="FineTuneLLM API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatMessage(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"  # Allow session tracking

class MessageResponse(BaseModel):
    id: str
    user_prompt: str
    model_response: str
    timestamp: str

@app.get("/")
async def root():
    return {
        "status": "online",
        "model_loaded": llm_model is not None,
        "history_file": str(HISTORY_FILE)
    }

@app.get("/status")
async def get_status():
    return {
        "model_loaded": llm_model is not None,
        "history_file": str(HISTORY_FILE)
    }

def format_chat_prompt(history: List[Dict[str, str]], new_message: str) -> str:
    """
    Format conversation history into a proper chat prompt.
    This helps the model understand the context and provide coherent responses.
    """
    # System instruction to guide the model's behavior - strengthened to prevent rambling
    prompt = """You are a helpful AI assistant. Provide clear, concise, and relevant responses.
Answer questions directly and briefly. Keep responses focused and under 3 sentences unless specifically asked for more detail.
Do not generate examples, puzzles, or unrelated content. Stay on topic.

"""
    
    # Add conversation history
    for msg in history:
        if msg["role"] == "user":
            prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n"
    
    # Add current user message
    prompt += f"User: {new_message}\nAssistant:"
    
    return prompt

@app.get("/messages", response_model=List[MessageResponse])
async def get_messages(limit: int = 20):
    """Get last N messages from history file"""
    try:
        if not HISTORY_FILE.exists():
            return []
        
        messages = []
        async with aiofiles.open(HISTORY_FILE, 'r') as f:
            content = await f.read()
            lines = content.strip().split('\n')
            
            # Parse messages from history file
            # Format: timestamp|session_id|role|content
            # Note: Content may contain pipe characters, so we split with maxsplit=3
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    parts = line.split('|', 3)
                    if len(parts) != 4:
                        logger.warning(f"[bold yellow]⚠ Malformed line {line_num}: expected 4 parts, got {len(parts)}[/bold yellow]", extra={"markup": True})
                        continue
                    
                    timestamp, session_id, role, content = parts
                    
                    # Group user and assistant messages together
                    if role == "user":
                        # This is a user message, create a new message entry
                        messages.append({
                            "timestamp": timestamp,
                            "session_id": session_id,
                            "user_prompt": content,
                            "model_response": ""
                        })
                    elif role == "assistant":
                        # This is an assistant response
                        # Try to match with the most recent user message without a response
                        if messages and messages[-1]["model_response"] == "":
                            messages[-1]["model_response"] = content
                        else:
                            # Orphaned assistant message - log a warning
                            logger.warning(f"[bold yellow]⚠ Orphaned assistant message at line {line_num}[/bold yellow]", extra={"markup": True})
                    else:
                        logger.warning(f"[bold yellow]⚠ Unknown role '{role}' at line {line_num}[/bold yellow]", extra={"markup": True})
                        
                except Exception as e:
                    logger.warning(f"[bold yellow]⚠ Could not parse line {line_num}: {str(e)}[/bold yellow]", extra={"markup": True})
                    continue
        
        # Get last N complete messages (where both user and assistant are present)
        complete_messages = [msg for msg in messages if msg["user_prompt"] and msg["model_response"]]
        limited_messages = complete_messages[-limit:] if len(complete_messages) > limit else complete_messages
        
        return [
            MessageResponse(
                id=f"{msg['timestamp']}_{msg['session_id']}",
                user_prompt=msg["user_prompt"],
                model_response=msg["model_response"],
                timestamp=msg["timestamp"]
            )
            for msg in limited_messages
        ]
    except Exception as e:
        logger.error(f"[bold red]Error fetching messages: {e}[/bold red]", extra={"markup": True})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
async def clear_history(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in conversation_history:
        conversation_history[session_id].clear()
        logger.info(f"[bold cyan]Cleared conversation history for session: {session_id}[/bold cyan]", extra={"markup": True})
    return {"status": "success", "message": f"Conversation history cleared for session: {session_id}"}

@app.post("/chat")
async def chat(message: ChatMessage):
    """Stream chat response token by token with conversation context"""
    if not llm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    user_prompt = message.prompt
    session_id = message.session_id or "default"
    
    logger.info(f"[bold magenta]User Prompt:[/bold magenta] [cyan]{user_prompt}[/cyan]", extra={"markup": True})
    logger.info(f"[bold magenta]Session ID:[/bold magenta] [cyan]{session_id}[/cyan]", extra={"markup": True})
    
    # Get conversation history for this session
    history = conversation_history[session_id]
    
    # Format the prompt with conversation context
    formatted_prompt = format_chat_prompt(history, user_prompt)
    
    logger.info(f"[bold cyan]Formatted prompt length:[/bold cyan] [white]{len(formatted_prompt)} chars[/white]", extra={"markup": True})
    
    async def generate():
        full_response = ""
        start_time = time.time()
        token_count = 0
        
        try:
            # Stream tokens with improved parameters to match LM Studio behavior
            for output in llm_model(
                formatted_prompt,
                max_tokens=150,  # Reduced from 512 to prevent rambling
                temperature=0.5,  # Reduced from 0.7 for more focused responses
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,  # Prevent repetition
                stream=True,
                # Enhanced stop sequences to prevent hallucination
                stop=[
                    "</s>", 
                    "User:", "\nUser:", 
                    "Human:", "\nHuman:",
                    "Question:", "\nQuestion:",
                    "\n\n\n",  # Stop on multiple newlines
                    "Example:",  # Stop before generating examples
                    "Here's an",  # Stop before "Here's an example"
                    "Let me",  # Stop before "Let me give you an example"
                ]
            ):
                if "choices" in output and len(output["choices"]) > 0:
                    token = output["choices"][0].get("text", "")
                    if token:
                        full_response += token
                        token_count += 1
                        yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Send completion signal
            elapsed_time = time.time() - start_time
            tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
            # Clean up the response (remove any trailing whitespace)
            full_response = full_response.strip()
            
            # Validate and truncate if response is too long (safety check)
            if len(full_response) > 800:  # characters
                logger.warning(f"[bold yellow]⚠ Response too long ({len(full_response)} chars), truncating...[/bold yellow]", extra={"markup": True})
                # Truncate at sentence boundary
                sentences = full_response.split('. ')
                if len(sentences) >= 3:
                    # Keep first 3 sentences and add period only if last sentence doesn't end with punctuation
                    truncated = '. '.join(sentences[:3])
                    if not truncated.endswith(('.', '!', '?')):
                        truncated += '.'
                    full_response = truncated
                # If fewer than 3 sentences, just truncate at 800 characters
                else:
                    full_response = full_response[:800].rstrip()
            
            logger.info(f"[bold green]Model Response:[/bold green] [yellow]{full_response}[/yellow]", extra={"markup": True})
            logger.info(f"[bold blue]Response Time:[/bold blue] [white]{elapsed_time:.2f}s[/white]", extra={"markup": True})
            logger.info(f"[bold blue]Tokens Generated:[/bold blue] [white]{token_count} ({tokens_per_sec:.2f} tokens/s)[/white]", extra={"markup": True})
            
            # Update conversation history
            conversation_history[session_id].append({"role": "user", "content": user_prompt})
            conversation_history[session_id].append({"role": "assistant", "content": full_response})
            
            # Keep only last 10 exchanges (20 messages) to prevent context from getting too long
            if len(conversation_history[session_id]) > 20:
                conversation_history[session_id] = conversation_history[session_id][-20:]
            
            # Save to history file
            try:
                user_timestamp = datetime.utcnow().isoformat()
                assistant_timestamp = datetime.utcnow().isoformat()
                async with aiofiles.open(HISTORY_FILE, 'a') as f:
                    # Write user message
                    await f.write(f"{user_timestamp}|{session_id}|user|{user_prompt}\n")
                    # Write assistant response
                    await f.write(f"{assistant_timestamp}|{session_id}|assistant|{full_response}\n")
                logger.info("[bold green]✓ Conversation saved to history.txt[/bold green]", extra={"markup": True})
            except Exception as e:
                logger.warning(f"[bold yellow]⚠ Could not save to history file: {e}[/bold yellow]", extra={"markup": True})
        
        except Exception as e:
            logger.error(f"[bold red]Error generating response: {e}[/bold red]", extra={"markup": True})
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
