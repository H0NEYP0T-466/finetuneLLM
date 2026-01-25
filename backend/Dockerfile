FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your backend code
COPY . /app

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8002

# Command to run your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

