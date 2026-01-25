#!/bin/bash
# Start script for the backend server
# Make sure MongoDB is running before starting this

echo "Starting FineTuneLLM Backend..."
cd "$(dirname "$0")/app"
python main.py
