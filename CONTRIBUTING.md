# 🤝 Contributing to FineTuneLLM

First off, thank you for considering contributing to FineTuneLLM! It's people like you that make this project great.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Commit Message Format](#commit-message-format)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

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

- Node.js 18+
- Python 3.9+
- MongoDB (for full stack development)

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
cd app && python main.py
```

### Docker Setup (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
```

## 💡 How to Contribute

### Reporting Bugs

- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Search existing issues first to avoid duplicates
- Include detailed steps to reproduce
- Provide system information (OS, Python version, Node version)

### Suggesting Features

- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternatives you've thought about

### Code Contributions

1. **Find an issue** to work on or create one
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** following our naming conventions
4. **Make your changes** with clear, focused commits
5. **Test your changes** thoroughly
6. **Submit a pull request**

## 🌿 Branch Naming Conventions

Use the following prefixes for your branches:

- `feat/` - New features
  - Example: `feat/add-model-caching`
- `fix/` - Bug fixes
  - Example: `fix/mongodb-connection-error`
- `docs/` - Documentation changes
  - Example: `docs/update-api-docs`
- `refactor/` - Code refactoring
  - Example: `refactor/simplify-chat-component`
- `test/` - Adding or updating tests
  - Example: `test/add-api-integration-tests`
- `chore/` - Maintenance tasks
  - Example: `chore/update-dependencies`
- `perf/` - Performance improvements
  - Example: `perf/optimize-token-streaming`

## 📝 Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```
feat(frontend): add token streaming animation

Implemented smooth token-by-token display with fade-in effect
for better user experience.

Closes #42
```

```
fix(backend): resolve MongoDB connection timeout

- Increased connection timeout to 30 seconds
- Added retry logic with exponential backoff
- Improved error logging

Fixes #38
```

## 🔄 Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Update the README.md** if needed
3. **Follow the PR template** when creating your pull request
4. **Link related issues** using keywords (Closes #123, Fixes #456)
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally
7. **Squash commits** if requested before merging

### PR Checklist

Before submitting, ensure:

- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated and passing
- [ ] All existing tests pass

## 🎨 Code Style Guidelines

### Frontend (TypeScript/React)

- Follow the existing ESLint configuration (see `eslint.config.js`)
- Use TypeScript for type safety
- Follow React best practices and hooks guidelines
- Use functional components with hooks
- Keep components small and focused
- Use meaningful variable and function names

**Linting:**
```bash
npm run lint
```

### Backend (Python)

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and modular
- Handle errors appropriately

**Format checking:**
```bash
# Install black and flake8
pip install black flake8

# Format code
black backend/

# Check style
flake8 backend/
```

### General Guidelines

- Write clear, self-documenting code
- Add comments for complex logic
- Keep line length reasonable (80-120 characters)
- Use consistent naming conventions
- Avoid magic numbers - use named constants

## ✅ Testing Requirements

### Frontend Tests

```bash
# Run frontend tests
npm test
```

### Backend Tests

```bash
# Run backend tests
cd backend
pytest

# Run specific test
pytest test_api.py -v

# Run with coverage
pytest --cov=app
```

### Integration Tests

```bash
# Run integration tests
./test_integration.sh
```

### Writing Tests

- Write tests for new features
- Update tests when modifying existing code
- Aim for meaningful test coverage
- Test edge cases and error conditions
- Use descriptive test names

## 🔍 Code Review Process

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check for:

- Code quality and style
- Test coverage
- Documentation updates
- Potential bugs or security issues
- Performance considerations

## 📞 Getting Help

- **Questions?** Open a [Discussion](https://github.com/H0NEYP0T-466/finetuneLLM/discussions)
- **Stuck?** Comment on the issue you're working on
- **Need clarification?** Ask in your pull request

## 🎉 Recognition

Contributors will be recognized in:
- GitHub's contributors page
- Release notes (for significant contributions)
- Project documentation (for major features)

Thank you for contributing to FineTuneLLM! 🚀
