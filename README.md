# FineTuneLLM - Terminal Style Chat Interface

A minimalistic terminal-style chat UI with React+TypeScript frontend and FastAPI backend for local LLM inference.

## Features

- 🖥️ Clean terminal-style chat interface with #111 background
- 🚀 FastAPI backend with local LLM support (GGUF format)
- 📊 Real-time token streaming
- 💾 MongoDB chat history storage
- 🎨 Rich and colorful server logs
- ⚡ Auto-loads last 20 messages on startup
- 🎓 **NEW: Fine-tune Phi-2 on custom datasets** ([Quick Start](COLAB_QUICKSTART.md) | [Full Guide](finetune.md))

## Prerequisites

- Node.js 18+
- Python 3.9+
- MongoDB (running locally on port 27017)

## Project Structure

```
finetuneLLM/
├── src/                    # Frontend source
│   ├── components/         # React components
│   ├── services/          # API services
│   ├── types/             # TypeScript types
│   └── styles/            # CSS styles
├── backend/
│   ├── app/
│   │   └── main.py        # FastAPI application
│   ├── model/             # Place .gguf model files here
│   └── requirements.txt   # Python dependencies
└── README.md
```

## Setup Instructions

### Option 1: Using Docker Compose (Recommended)

This is the easiest way to run both the backend and MongoDB:

1. Place your GGUF model file in `backend/model/` directory

2. Start all services:
```bash
docker-compose up -d
```

3. Check logs:
```bash
docker-compose logs -f backend
```

4. Stop services:
```bash
docker-compose down
```

The backend will run on `http://localhost:8002` and MongoDB on `localhost:27017`.

**Note:** Using docker-compose ensures:
- ✅ MongoDB is properly connected (no connection refused errors)
- ✅ Database `finetuneLLM` is automatically created
- ✅ Messages are persisted across restarts
- ✅ Proper networking between services

### Option 2: Manual Setup

#### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. (Optional) Configure environment variables:
```bash
cd backend
cp .env.example .env
# Edit .env to customize MongoDB URI and CORS settings
```

3. Place your GGUF model file in `backend/model/` directory

4. Ensure MongoDB is running:
```bash
# On Linux/Mac
sudo systemctl start mongod

# Or with Docker
docker run -d -p 27017:27017 --name mongodb mongo
```

5. Start the backend server:
```bash
cd backend
./start_server.sh
# Or manually:
# cd app && python main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. (Optional) Configure environment variables:
```bash
cp .env.example .env
# Edit .env to customize API URL if needed
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:5173`

## Usage

1. Start MongoDB
2. Start the backend server (it will load the LLM model)
3. Start the frontend development server
4. Open your browser to `http://localhost:5173`
5. Start chatting!

## Features in Detail

### Backend
- Loads GGUF models on server startup
- Streams tokens in real-time using Server-Sent Events
- Logs user prompts and model responses with rich formatting
- Tracks and logs response times
- Stores all conversations in MongoDB

### Frontend
- Terminal-style UI with green text on black background (#111)
- Token-by-token streaming display
- Loading indicators during model initialization
- Auto-scroll to latest messages
- Loads last 20 messages from database on startup

## API Endpoints

- `GET /` - Health check
- `GET /status` - Check model and database status
- `GET /messages?limit=20` - Get last N messages
- `POST /chat` - Send message and stream response

## Development

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Fine-Tuning

Want to fine-tune the Phi-2 model on your own data? We've got you covered!

### 🚀 Quick Start

1. Prepare your dataset as `dataset.xlsx` (Excel file with Q&A pairs)
2. Upload to Google Colab and run `finetuneCollab.py`
3. Get your fine-tuned model in 20-30 minutes!

**Read the guides:**
- [Colab Quick Start](COLAB_QUICKSTART.md) - Get started in 5 minutes
- [Complete Guide](finetune.md) - Full technical documentation
- [Feature Overview](FINETUNE_README.md) - What's included

### What You Get

- ✅ Parameter-efficient fine-tuning with LoRA
- ✅ Runs on free Google Colab GPU
- ✅ Automatic training visualizations
- ✅ Complete documentation
- ✅ Example dataset included

### Files

- `finetuneCollab.py` - Main training script
- `finetune.md` - Technical documentation (25KB)
- `COLAB_QUICKSTART.md` - Quick start guide
- `requirements-finetune.txt` - Dependencies
- `dataset_example.xlsx` - Sample data

## License

MIT
