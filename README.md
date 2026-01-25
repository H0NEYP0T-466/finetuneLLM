# FineTuneLLM - Terminal Style Chat Interface

A minimalistic terminal-style chat UI with React+TypeScript frontend and FastAPI backend for local LLM inference.

## Features

- 🖥️ Clean terminal-style chat interface with #111 background
- 🚀 FastAPI backend with local LLM support (GGUF format)
- 📊 Real-time token streaming
- 💾 MongoDB chat history storage
- 🎨 Rich and colorful server logs
- ⚡ Auto-loads last 20 messages on startup

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

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Place your GGUF model file in `backend/model/` directory

3. Ensure MongoDB is running:
```bash
# On Linux/Mac
sudo systemctl start mongod

# Or with Docker
docker run -d -p 27017:27017 --name mongodb mongo
```

4. Start the backend server:
```bash
cd backend/app
python main.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
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

## License

MIT
