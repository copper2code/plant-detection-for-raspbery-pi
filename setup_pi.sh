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
echo "[4/5] Installing Python requirements..."
source venv/bin/activate
pip install --upgrade pip
# Set index-url mapping for piwheels just in case it helps find pre-compiled packages on Raspberry Pi
pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt

# 5. Setup systemd service for running at boot
echo "[5/5] Setting up systemd service to run at boot..."
PROJECT_DIR="$(pwd)"
CURRENT_USER="$USER"

cat <<EOF | sudo tee /etc/systemd/system/plantcare.service > /dev/null
[Unit]
Description=PlantCare Pi Web Server
After=network-online.target
Wants=network-online.target

[Service]
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python app/app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable plantcare.service --now

echo "================================================="
echo "Setup Complete! 🎉"
echo "The application has been installed as a background service."
echo "It will automatically start whenever the Raspberry Pi boots."
echo ""
echo "To check the status of the service:"
echo "  sudo systemctl status plantcare.service"
echo ""
echo "To view live logs:"
echo "  sudo journalctl -u plantcare.service -f"
echo ""
echo "Open a browser to http://<YOUR_PI_IP>:5000"
echo "================================================="
