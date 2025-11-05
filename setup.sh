#!/bin/bash

echo "Setting up Voice Phishing Detection Project..."

# Create virtual environment
python3.9 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Run: source venv/bin/activate"
