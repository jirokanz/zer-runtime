#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
echo "Installation complete. Activate with: source venv/bin/activate"
