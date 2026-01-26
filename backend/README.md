# FineTuneLLM Backend

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your GGUF model file in the `model/` directory

3. (Optional) Make sure MongoDB is running locally on port 27017
   - MongoDB is optional - the service will work without it
   - Without MongoDB, chat history won't be persisted to database

4. Start the server:
```bash
cd app
python main.py
```

The server will run on `http://localhost:8002`

## Features

### Conversation Context Management
- Maintains conversation history per session
- Proper chat template formatting with system instructions
- Context-aware responses that reference previous messages
- Automatic context window management (keeps last 10 exchanges)

### Anti-Repetition Mechanisms
- Repeat penalty to prevent token loops
- Top-K sampling for better token selection
- Improved stop sequences to end responses appropriately

### Session Support
- Multiple concurrent chat sessions
- Session-based conversation history
- `/clear-history` endpoint to reset conversation

## API Endpoints

- `GET /` - Health check
- `GET /status` - Get model and database status
- `GET /messages?limit=20` - Get last N messages (requires MongoDB)
- `POST /chat` - Stream chat response (Server-Sent Events)
  - Body: `{"prompt": "your message", "session_id": "optional-session-id"}`
  - Default session_id: "default"
- `POST /clear-history?session_id=default` - Clear conversation history for a session

## Configuration

### Environment Variables
- `MONGODB_URI` - MongoDB connection string (default: `mongodb://localhost:27017/`)

### Model Parameters
The model is configured with conservative settings to match LM Studio behavior:
- Context window: 2048 tokens
- Threads: 4
- GPU layers: -1 (use all GPU if available)
- Repeat penalty: 1.1
- Temperature: 0.5 (reduced for more focused responses)
- Top-p: 0.9
- Top-k: 40
- Max tokens per response: 150 (prevents rambling)

### Stop Sequences
Enhanced stop sequences to prevent hallucination:
- `</s>`, `User:`, `\nUser:`, `Human:`, `\nHuman:`
- `Question:`, `\nQuestion:`
- `\n\n\n` (multiple newlines)
- `Example:`, `Here's an`, `Let me` (prevents example generation)

## Docker Deployment

### Using Docker Compose (Recommended)

From the root directory, use docker-compose to run both backend and MongoDB:

```bash
docker-compose up -d
```

This automatically:
- ✅ Starts MongoDB with proper networking
- ✅ Creates the `finetuneLLM` database
- ✅ Connects backend to MongoDB (no connection errors)
- ✅ Persists data in a Docker volume

### Manual Docker Deployment

When running in Docker manually:
1. Ensure the model file is mounted to `/app/model/`
2. MongoDB is optional - service works without it
3. Set `MONGODB_URI` environment variable if using external MongoDB
   - Example: `mongodb://mongodb:27017/` when using docker network
4. CPU usage will spike to 100% during generation (this is normal)
5. RAM usage depends on model size (typically 4-8GB for 7B models)

**Important:** When running backend in Docker, `localhost:27017` refers to the container itself,
not your host machine. Use docker-compose or set `MONGODB_URI=mongodb://host.docker.internal:27017/`
to connect to MongoDB on the host.

## Troubleshooting

See `issue.txt` in the root directory for detailed information about:
- Common issues and their root causes
- Performance optimization
- Chat quality improvements
- Docker deployment considerations

