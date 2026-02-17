# 🤝 Contributing to FineTuneLLM

Thank you for your interest in contributing to FineTuneLLM! We welcome contributions from the community.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Branch Naming Convention](#branch-naming-convention)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)

## 🚀 Getting Started

1. **Fork the Repository**
   - Click the "Fork" button at the top right of the repository page
   - Clone your fork locally:
     ```bash
     git clone https://github.com/YOUR_USERNAME/finetuneLLM.git
     cd finetuneLLM
     ```

2. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/H0NEYP0T-466/finetuneLLM.git
   ```

3. **Keep Your Fork Synced**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

## 🛠 Development Setup

### Frontend Setup

1. Install Node.js dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Run linter:
   ```bash
   npm run lint
   ```

### Backend Setup

1. Create a virtual environment:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the backend server:
   ```bash
   ./start_server.sh
   ```

### Database Setup

- Ensure MongoDB is running on `localhost:27017`
- Or use Docker Compose:
  ```bash
  docker-compose up -d
  ```

## 💡 How to Contribute

### Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Node/Python versions)

### Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) and describe:
- The problem your feature solves
- Your proposed solution
- Alternative approaches considered

### Code Contributions

1. Check existing issues or create a new one
2. Comment on the issue to let others know you're working on it
3. Follow the development workflow below

## 🌿 Branch Naming Convention

Use descriptive branch names with prefixes:

- `feat/` - New features
  - Example: `feat/add-model-switching`
- `fix/` - Bug fixes
  - Example: `fix/mongodb-connection-error`
- `docs/` - Documentation updates
  - Example: `docs/update-api-guide`
- `refactor/` - Code refactoring
  - Example: `refactor/simplify-streaming-logic`
- `test/` - Adding or updating tests
  - Example: `test/add-backend-unit-tests`
- `chore/` - Maintenance tasks
  - Example: `chore/update-dependencies`

## 📝 Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(frontend): add dark mode toggle

Add a toggle button to switch between light and dark themes.
Preserves user preference in localStorage.

Closes #123
```

```
fix(backend): resolve MongoDB connection timeout

Increase connection timeout from 5s to 30s to handle slower networks.
Add retry logic with exponential backoff.

Fixes #456
```

## 🔄 Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments where necessary

3. **Test Your Changes**
   - Run linters: `npm run lint`
   - Test manually in the browser
   - Verify backend endpoints work correctly

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature"
   ```

5. **Push to Your Fork**
   ```bash
   git push origin feat/your-feature-name
   ```

6. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Link related issues

7. **Address Review Comments**
   - Make requested changes
   - Push updates to the same branch
   - Respond to reviewer feedback

8. **Wait for Approval**
   - At least one maintainer must approve
   - All CI checks must pass
   - No merge conflicts

## 🎨 Code Style Guidelines

### TypeScript/JavaScript

- Use TypeScript for type safety
- Follow existing ESLint configuration (see `eslint.config.js`)
- Use functional components and hooks
- Prefer `const` over `let`
- Use meaningful variable names
- Keep functions small and focused

### Python

- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for functions and classes
- Keep line length under 88 characters (Black formatter standard)
- Use async/await for I/O operations

### General

- Write self-documenting code
- Add comments only when necessary to explain "why", not "what"
- Remove commented-out code
- Keep files focused on a single responsibility

## ✅ Testing Requirements

### Frontend

- Manually test all UI changes in the browser
- Verify responsive design works on different screen sizes
- Test with different browsers if possible
- Ensure no console errors

### Backend

- Test all API endpoints manually or with tools like `curl` or Postman
- Verify error handling works correctly
- Test with different model configurations
- Check MongoDB operations work as expected

### Integration

- Test the complete flow from frontend to backend
- Verify chat messages are stored and retrieved correctly
- Test streaming functionality
- Ensure proper error messages are displayed

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## ❓ Questions?

- Open a [Discussion](https://github.com/H0NEYP0T-466/finetuneLLM/discussions)
- Check existing [Issues](https://github.com/H0NEYP0T-466/finetuneLLM/issues)
- Read the [Documentation](README.md)

## 🙏 Thank You!

Your contributions help make FineTuneLLM better for everyone!
