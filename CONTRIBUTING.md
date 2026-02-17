# 🤝 Contributing to FineTuneLLM

Thank you for your interest in contributing to FineTuneLLM! We welcome contributions from the community.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)

## 🚀 Getting Started

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/finetuneLLM.git
   cd finetuneLLM
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/H0NEYP0T-466/finetuneLLM.git
   ```

## 🛠 Development Setup

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- MongoDB (or use Docker Compose)
- Git

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup

```bash
# Create virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your GGUF model in backend/model/

# Start backend server
cd app
python main.py
```

### Using Docker Compose (Recommended)

```bash
# Start all services (backend + MongoDB)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## 🌿 Branch Naming Conventions

Use the following prefixes for your branches:

- `feat/` - New features (e.g., `feat/add-conversation-export`)
- `fix/` - Bug fixes (e.g., `fix/streaming-token-display`)
- `docs/` - Documentation changes (e.g., `docs/update-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/api-client`)
- `test/` - Adding or updating tests (e.g., `test/chat-component`)
- `chore/` - Maintenance tasks (e.g., `chore/update-dependencies`)
- `perf/` - Performance improvements (e.g., `perf/optimize-model-loading`)

**Example:**
```bash
git checkout -b feat/add-dark-mode
```

## 💬 Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (whitespace, formatting)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to build process or auxiliary tools

### Examples

```bash
feat(chat): add message export functionality

fix(backend): resolve model loading timeout issue

docs(readme): update installation instructions

refactor(api): simplify chat endpoint logic
```

## 🔄 Pull Request Process

1. **Update from upstream** before starting work:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a new branch** following our naming conventions

3. **Make your changes** following the code style guidelines

4. **Test your changes** thoroughly (see [Testing Requirements](#testing-requirements))

5. **Commit your changes** using conventional commit messages

6. **Push to your fork**:
   ```bash
   git push origin your-branch-name
   ```

7. **Open a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Reference any related issues (e.g., "Closes #123")
   - Add screenshots for UI changes
   - Request review from maintainers

8. **Address review feedback** promptly and professionally

9. **Ensure CI checks pass** before merging

## 🎨 Code Style Guidelines

### TypeScript/React (Frontend)

- Follow the existing ESLint configuration
- Run linter before committing:
  ```bash
  npm run lint
  ```
- Use TypeScript types, avoid `any`
- Use functional components with hooks
- Keep components small and focused
- Use meaningful variable and function names

### Python (Backend)

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Keep functions focused and modular
- Add docstrings for complex functions
- Use meaningful variable names

### General

- Write self-documenting code
- Add comments only when necessary to explain "why", not "what"
- Keep lines under 100 characters when practical
- Use consistent indentation (2 spaces for JS/TS, 4 spaces for Python)

## ✅ Testing Requirements

### Before Submitting

1. **Manual Testing**
   - Test your changes locally
   - Verify both frontend and backend work together
   - Test edge cases and error conditions

2. **Linting**
   ```bash
   # Frontend
   npm run lint
   
   # Backend (if using tools like flake8)
   flake8 backend/app
   ```

3. **Build Test**
   ```bash
   # Frontend
   npm run build
   ```

4. **Integration Testing**
   - Ensure MongoDB connection works
   - Test chat functionality end-to-end
   - Verify model loading and inference

### Writing Tests

- Add tests for new features when applicable
- Update existing tests if you modify behavior
- Ensure tests are clear and maintainable

## 🆘 Getting Help

- 💬 [GitHub Discussions](https://github.com/H0NEYP0T-466/finetuneLLM/discussions) - Ask questions
- 🐛 [Issue Tracker](https://github.com/H0NEYP0T-466/finetuneLLM/issues) - Report bugs
- 📧 Contact maintainers if you need additional guidance

## 📜 Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## 🙏 Recognition

All contributors will be recognized in our project. Thank you for making FineTuneLLM better!

---

**Questions?** Feel free to open an issue or start a discussion. We're here to help! 🎉
