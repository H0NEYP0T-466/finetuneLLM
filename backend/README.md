# FineTuneLLM Backend

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your GGUF model file in the `model/` directory

3. Make sure MongoDB is running locally on port 27017

4. Start the server:
```bash
cd app
python main.py
```

The server will run on `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /status` - Get model and database status
- `GET /messages?limit=20` - Get last N messages
- `POST /chat` - Stream chat response (Server-Sent Events)
