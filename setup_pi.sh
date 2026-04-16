#!/bin/bash
# PlantCare Pi Setup Script

set -e

echo "================================================="
echo "   PlantCare Pi Setup Script"
echo "================================================="
echo "This script will install system dependencies,"
echo "create a virtual environment, and install PyTorch"
echo "and OpenCV for this project."
echo ""

cd "$(dirname "$0")"

# 1. Update package lists
echo "[1/4] Updating package lists..."
sudo apt-get update

# 2. Install system dependencies for OpenCV and building python packages
echo "[2/4] Installing system dependencies..."
sudo apt-get install -y python3-venv python3-dev \
    libopenblas-dev libgl1-mesa-glx libglib2.0-0 \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libqt5gui5 libqt5webkit5 libqt5test5 libqt5core5a

# 3. Create Virtual Environment
echo "[3/4] Creating Python virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# 4. Install Requirements
echo "[4/4] Installing Python requirements..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "================================================="
echo "Setup Complete! 🎉"
echo "To start the application:"
echo ""
echo "  source venv/bin/activate"
echo "  python app/app.py"
echo ""
echo "Then open a browser to http://<YOUR_PI_IP>:5000"
echo "================================================="
