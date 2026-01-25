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
from typing import Optional, List
import json
from datetime import datetime

# Setup Rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FineTuneLLM API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm_model: Optional[Llama] = None
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
chat_collection = None

# Models
class ChatMessage(BaseModel):
    prompt: str

class MessageResponse(BaseModel):
    id: str
    user_prompt: str
    model_response: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global llm_model, mongo_client, db, chat_collection
    
    # Connect to MongoDB
    try:
        logger.info("[bold cyan]Connecting to MongoDB...[/bold cyan]", extra={"markup": True})
        mongo_client = AsyncIOMotorClient("mongodb://localhost:27017/")
        db = mongo_client["finetuneLLM"]
        chat_collection = db["chats"]
        logger.info("[bold green]✓ MongoDB connected successfully[/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"[bold red]✗ MongoDB connection failed: {e}[/bold red]", extra={"markup": True})
    
    # Load LLM model
    model_path = Path(__file__).parent.parent / "model"
    gguf_files = list(model_path.glob("*.gguf"))
    
    if not gguf_files:
        logger.warning("[bold yellow]⚠ No GGUF model files found in model directory[/bold yellow]", extra={"markup": True})
        logger.warning("[bold yellow]Please place a .gguf model file in the backend/model directory[/bold yellow]", extra={"markup": True})
        return
    
    model_file = gguf_files[0]
    logger.info(f"[bold cyan]Loading model: {model_file.name}[/bold cyan]", extra={"markup": True})
    
    start_time = time.time()
    try:
        llm_model = Llama(
            model_path=str(model_file),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False
        )
        load_time = time.time() - start_time
        logger.info(f"[bold green]✓ Model loaded successfully in {load_time:.2f} seconds[/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"[bold red]✗ Model loading failed: {e}[/bold red]", extra={"markup": True})

@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("[bold cyan]MongoDB connection closed[/bold cyan]", extra={"markup": True})

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

@app.get("/messages", response_model=List[MessageResponse])
async def get_messages(limit: int = 20):
    """Get last N messages from database"""
    if not chat_collection:
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

@app.post("/chat")
async def chat(message: ChatMessage):
    """Stream chat response token by token"""
    if not llm_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    prompt = message.prompt
    logger.info(f"[bold magenta]User Prompt:[/bold magenta] [cyan]{prompt}[/cyan]", extra={"markup": True})
    
    async def generate():
        full_response = ""
        start_time = time.time()
        
        try:
            # Stream tokens
            for output in llm_model(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stream=True,
                stop=["</s>", "User:", "\n\n"]
            ):
                if "choices" in output and len(output["choices"]) > 0:
                    token = output["choices"][0].get("text", "")
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Send completion signal
            elapsed_time = time.time() - start_time
            yield f"data: {json.dumps({'done': True})}\n\n"
            
            logger.info(f"[bold green]Model Response:[/bold green] [yellow]{full_response}[/yellow]", extra={"markup": True})
            logger.info(f"[bold blue]Response Time:[/bold blue] [white]{elapsed_time:.2f}s[/white]", extra={"markup": True})
            
            # Save to database
            if chat_collection:
                try:
                    await chat_collection.insert_one({
                        "user_prompt": prompt,
                        "model_response": full_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "response_time": elapsed_time
                    })
                except Exception as e:
                    logger.error(f"[bold red]Error saving to database: {e}[/bold red]", extra={"markup": True})
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
