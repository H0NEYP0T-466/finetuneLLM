from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
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
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
chat_collection = None

# Store conversation history per session (in-memory for simplicity)
# Format: {session_id: [{"role": "user/assistant", "content": "..."}]}
conversation_history: Dict[str, List[Dict[str, str]]] = defaultdict(list)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global llm_model, mongo_client, db, chat_collection
    
    # Connect to MongoDB
    try:
        logger.info("[bold cyan]Connecting to MongoDB...[/bold cyan]", extra={"markup": True})
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        mongo_client = AsyncIOMotorClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("[bold green]✓ MongoDB connection successful[/bold green]", extra={"markup": True})
        
        # Check if database exists, create if it doesn't
        db_list = await mongo_client.list_database_names()
        if "finetuneLLM" not in db_list:
            logger.info("[bold cyan]Creating database: finetuneLLM[/bold cyan]", extra={"markup": True})
        else:
            logger.info("[bold cyan]Database finetuneLLM already exists[/bold cyan]", extra={"markup": True})
        
        # Connect to database and collection
        db = mongo_client["finetuneLLM"]
        chat_collection = db["chats"]
        
        # Ensure the collection exists by attempting an operation
        await chat_collection.count_documents({})
        logger.info("[bold green]✓ Database and collection ready[/bold green]", extra={"markup": True})
    except Exception as e:
        logger.warning(f"[bold yellow]⚠ MongoDB connection failed: {e}[/bold yellow]", extra={"markup": True})
        logger.warning("[bold yellow]⚠ Continuing without database - chats will not be saved[/bold yellow]", extra={"markup": True})
        mongo_client = None
        db = None
        chat_collection = None
    
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
    if mongo_client:
        mongo_client.close()
        logger.info("[bold cyan]MongoDB connection closed[/bold cyan]", extra={"markup": True})

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
        "database_connected": mongo_client is not None
    }

@app.get("/status")
async def get_status():
    return {
        "model_loaded": llm_model is not None,
        "database_connected": mongo_client is not None
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
    """Get last N messages from database"""
    if chat_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        cursor = chat_collection.find().sort("timestamp", -1).limit(limit)
        messages = await cursor.to_list(length=limit)
        
        # Reverse to get chronological order
        messages.reverse()
        
        return [
            MessageResponse(
                id=str(msg["_id"]),
                user_prompt=msg["user_prompt"],
                model_response=msg["model_response"],
                timestamp=msg["timestamp"]
            )
            for msg in messages
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
            
            # Save to database if available
            if chat_collection is not None:
                try:
                    await chat_collection.insert_one({
                        "session_id": session_id,
                        "user_prompt": user_prompt,
                        "model_response": full_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "response_time": elapsed_time,
                        "token_count": token_count,
                        "tokens_per_sec": tokens_per_sec
                    })
                except Exception as e:
                    logger.warning(f"[bold yellow]⚠ Could not save to database: {e}[/bold yellow]", extra={"markup": True})
        
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
