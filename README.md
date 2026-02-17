<h1 align="center">FineTuneLLM</h1>

<p align="center">
  <strong>Terminal Style Chat Interface with Fine-Tuning Capabilities</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/stars/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/github/forks/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Forks">
  <img src="https://img.shields.io/github/issues/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Issues">
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Commit Activity">
  <img src="https://img.shields.io/github/languages/top/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Top Language">
  <img src="https://img.shields.io/github/languages/count/H0NEYP0T-466/finetuneLLM?style=for-the-badge" alt="Languages">
</p>

<p align="center">
  A minimalistic terminal-style chat UI with React+TypeScript frontend and FastAPI backend for local LLM inference. 
  Features parameter-efficient fine-tuning with LoRA on custom datasets using Google Colab.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="https://github.com/H0NEYP0T-466/finetuneLLM/issues">Issues</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Project Structure](#-project-structure)
- [Setup Instructions](#-setup-instructions)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Dependencies](#-dependencies)
- [API Endpoints](#-api-endpoints)
- [Fine-Tuning](#-fine-tuning)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)

---

## ✨ Features

- 🖥️ Clean terminal-style chat interface with #111 background
- 🚀 FastAPI backend with local LLM support (GGUF format)
- 📊 Real-time token streaming
- 💾 MongoDB chat history storage
- 🎨 Rich and colorful server logs
- ⚡ Auto-loads last 20 messages on startup
- 🎓 **NEW: Fine-tune Phi-2 on custom datasets** ([Quick Start](COLAB_QUICKSTART.md) | [Full Guide](finetune.md))

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/H0NEYP0T-466/finetuneLLM.git
cd finetuneLLM

# 2. Place your GGUF model in backend/model/

# 3. Start with Docker Compose (easiest)
docker-compose up -d

# 4. Install frontend dependencies
npm install

# 5. Start frontend development server
npm run dev

# 6. Open http://localhost:5173 in your browser
```

---

## 📋 Prerequisites

- Node.js 18+
- Python 3.9+
- MongoDB (running locally on port 27017)
- GGUF format LLM model file

---

## 📂 Project Structure

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

---

## ⚙️ Setup Instructions

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

---

## 💻 Usage

1. Start MongoDB
2. Start the backend server (it will load the LLM model)
3. Start the frontend development server
4. Open your browser to `http://localhost:5173`
5. Start chatting!

### Features in Detail

**Backend**
- Loads GGUF models on server startup
- Streams tokens in real-time using Server-Sent Events
- Logs user prompts and model responses with rich formatting
- Tracks and logs response times
- Stores all conversations in MongoDB

**Frontend**
- Terminal-style UI with green text on black background (#111)
- Token-by-token streaming display
- Loading indicators during model initialization
- Auto-scroll to latest messages
- Loads last 20 messages from database on startup

---

## 🛠 Tech Stack

### Frontend
![React](https://img.shields.io/badge/React-19.2.0-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-7.2.4-646CFF?style=for-the-badge&logo=vite&logoColor=white)

### Backend
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Latest-47A248?style=for-the-badge&logo=mongodb&logoColor=white)

### ML/AI
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.48.0-FFD21E?style=for-the-badge)
![LLaMA](https://img.shields.io/badge/llama--cpp--python-0.3.2-000000?style=for-the-badge)

### DevOps
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## 📦 Dependencies

### Frontend Dependencies (Runtime)

![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&logo=react&color=61DAFB)
![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&logo=react&color=61DAFB)

### Frontend Dependencies (Development)

![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&logo=typescript&color=3178C6)
![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&logo=vite&color=646CFF)
![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&logo=eslint&color=4B32C3)

### Backend Dependencies (Runtime)

![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&logo=fastapi&color=009688)
![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn&color=2C5F9E)
![llama-cpp-python](https://img.shields.io/pypi/v/llama-cpp-python?style=for-the-badge&label=llama-cpp-python&color=000000)
![pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic&logo=pydantic&color=E92063)
![rich](https://img.shields.io/pypi/v/rich?style=for-the-badge&label=rich&color=orange)

### Fine-Tuning Dependencies

![transformers](https://img.shields.io/pypi/v/transformers?style=for-the-badge&label=transformers&logo=huggingface&color=FFD21E)
![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch&logo=pytorch&color=EE4C2C)
![peft](https://img.shields.io/pypi/v/peft?style=for-the-badge&label=peft&color=FFD21E)
![accelerate](https://img.shields.io/pypi/v/accelerate?style=for-the-badge&label=accelerate&color=FFD21E)
![datasets](https://img.shields.io/pypi/v/datasets?style=for-the-badge&label=datasets&logo=huggingface&color=FFD21E)

---

## 📡 API Endpoints

- `GET /` - Health check
- `GET /status` - Check model and database status
- `GET /messages?limit=20` - Get last N messages
- `POST /chat` - Send message and stream response

---

## 🔨 Development

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

Lint code:
```bash
npm run lint
```

---

## 🎓 Fine-Tuning

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

---

## 📚 Documentation

- [README.md](README.md) - Main documentation (this file)
- [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) - Quick start guide for fine-tuning
- [FINETUNE_README.md](FINETUNE_README.md) - Fine-tuning feature overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [finetune.md](finetune.md) - Complete fine-tuning documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy and vulnerability reporting
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community code of conduct

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to fork and clone the repository
- Development setup and workflow
- Branch naming conventions
- Commit message format
- Pull request process
- Code style guidelines

Before contributing, please read our [Code of Conduct](CODE_OF_CONDUCT.md).

**Quick Links:**
- [Report a Bug](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Request a Feature](.github/ISSUE_TEMPLATE/feature_request.yml)
- [Ask Questions](https://github.com/H0NEYP0T-466/finetuneLLM/discussions)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🛡 Security

Security is important to us. If you discover a security vulnerability, please follow our [Security Policy](SECURITY.md).

**Do not report security vulnerabilities through public GitHub issues.**

- Read our [Security Policy](SECURITY.md)
- Report vulnerabilities privately through GitHub Security Advisories
- Response time: Within 48 hours

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a>
</p>

<p align="center">
  <a href="#-table-of-contents">⬆ Back to Top</a>
</p>
