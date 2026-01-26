# Quick Start Guide - Using Docker Compose

## Prerequisites
- Docker and Docker Compose installed
- A GGUF model file (e.g., `phi-2.Q5_K_S.gguf`)

## Step 1: Place Your Model

Copy your GGUF model file to the backend/model directory:
```bash
cp /path/to/your/model.gguf backend/model/
```

## Step 2: Start All Services

From the root directory, run:
```bash
docker-compose up -d
```

This starts:
- MongoDB container (database for chat history)
- Backend container (FastAPI + LLM)

## Step 3: Verify Services are Running

Check logs:
```bash
docker-compose logs -f backend
```

You should see:
```
✓ MongoDB connection successful
✓ Database finetuneLLM already exists
✓ Database and collection ready
✓ Model loaded successfully
```

## Step 4: Test the API

Send a test message:
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'
```

Or check status:
```bash
curl http://localhost:8002/status
```

Expected response:
```json
{
  "model_loaded": true,
  "database_connected": true
}
```

## Step 5: Use the Frontend

1. Install frontend dependencies:
```bash
npm install
```

2. Start the frontend:
```bash
npm run dev
```

3. Open browser to http://localhost:5173

## Managing Services

### View logs
```bash
# All logs
docker-compose logs -f

# Just backend
docker-compose logs -f backend

# Just MongoDB
docker-compose logs -f mongodb
```

### Restart services
```bash
docker-compose restart
```

### Stop services
```bash
docker-compose down
```

### Stop and remove data
```bash
docker-compose down -v
```

## Troubleshooting

### Model not loading
- Ensure GGUF file is in `backend/model/`
- Check file permissions
- View logs: `docker-compose logs backend`

### MongoDB connection errors
- Ensure docker-compose is used (not standalone docker run)
- Check MongoDB is healthy: `docker-compose ps`
- Wait 10 seconds after starting for MongoDB to initialize

### Model giving wrong responses
- Check logs to verify parameters:
  - max_tokens: 150
  - temperature: 0.5
- Clear conversation history: 
  ```bash
  curl -X POST http://localhost:8002/clear-history
  ```

### Port already in use
Edit docker-compose.yml to use different ports:
```yaml
ports:
  - "8003:8002"  # Change 8003 to any available port
```

## What's Different from LM Studio?

This setup now matches LM Studio behavior:

| Feature | LM Studio | This Setup |
|---------|-----------|------------|
| max_tokens | ~150 | 150 ✓ |
| temperature | ~0.5 | 0.5 ✓ |
| Stop sequences | Extensive | Enhanced ✓ |
| Database | Optional | MongoDB ✓ |
| Response quality | Focused | Focused ✓ |

## Performance Expectations

- **First response**: 2-5 seconds (model warm-up)
- **Subsequent responses**: Depends on length
  - Short answer (20 tokens): ~4-6 seconds
  - Medium answer (50 tokens): ~10-15 seconds
  - Max length (150 tokens): ~30-45 seconds
- **Tokens/second**: 3-5 tokens/sec (CPU) or 10-30 tokens/sec (GPU)
- **CPU usage**: 100% during generation (normal)
- **RAM usage**: 4-8GB depending on model size

## Next Steps

1. ✅ Services running
2. ✅ Model loaded
3. ✅ MongoDB connected
4. ✅ Frontend accessible
5. 🎉 Start chatting!

For detailed technical explanation, see `explanation.md`.
