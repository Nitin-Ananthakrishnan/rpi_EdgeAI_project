#!/bin/bash

# --- MASTER DEPLOYMENT SCRIPT FOR EDGE AI ASSISTANT ---

PROJECT_DIR="$HOME/SignLanguageEdge"
LOG_FILE="$PROJECT_DIR/installation_report.txt"

echo "=================================================="
echo "   AUTONOMOUS EDGE AI: SYSTEM INITIALIZATION      "
echo "=================================================="

# Create project directory if it doesn't exist
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# --- PHASE 1: FIRST-TIME SETUP (Only runs if venv is missing) ---
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "[!] First time setup detected. Initializing installation..."
    echo "--- INSTALLATION AUDIT REPORT $(date) ---" > $LOG_FILE
    
    echo " -> Updating Linux OS..."
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update && sudo apt-get upgrade -y >> $LOG_FILE 2>&1

    echo " -> Installing Core System Dependencies (Camera & Audio)..."
    # Note: Using 'flite' for the offline voice engine
    APT_PKGS="flite alsa-utils v4l-utils python3-pip python3-venv"
    for pkg in $APT_PKGS; do
        if sudo apt-get install -y $pkg >> $LOG_FILE 2>&1; then
            echo "    [OK] $pkg"
        else
            echo "    [FAIL] $pkg - Check $LOG_FILE"
        fi
    done

    echo " -> Creating Python Virtual Environment..."
    python3 -m venv venv
    source venv/bin/activate

    echo " -> Installing Edge AI Python Libraries..."
    pip install --upgrade pip >> $LOG_FILE 2>&1
    PIP_LIBS="opencv-python-headless tflite-runtime streamlit psutil pandas matplotlib"
    for lib in $PIP_LIBS; do
        echo "    Installing $lib..."
        if pip install $lib >> $LOG_FILE 2>&1; then
            echo "    [OK] $lib"
        else
            echo "    [FAIL] $lib - Check $LOG_FILE"
        fi
    done

    echo " -> Setting Hardware Permissions (Audio/Video)..."
    sudo usermod -a -G video $USER
    sudo usermod -a -G audio $USER

    echo "[?] First-time setup completed successfully!"
    echo "--------------------------------------------------"
else
    echo "[?] System dependencies already installed."
fi


# --- PHASE 2: SYSTEM LAUNCH ---

echo "[*] Activating Virtual Environment..."
source venv/bin/activate

echo "[*] Configuring Audio Routing..."
# Prevents audio from trying to go through a disabled HDMI port
amixer cset numid=3 1 > /dev/null 2>&1 || true

echo "[*] Initializing Telemetry Logs..."
if [ ! -f "system_telemetry.txt" ]; then
    echo "System Boot: $(date)" > system_telemetry.txt
fi

echo "=================================================="
echo " ?? LAUNCHING TELEMETRY DASHBOARD...              "
echo " Open a browser and go to: http://localhost:8501  "
echo "=================================================="

# Add this inside start_system.sh before the Streamlit line:
echo "[*] Connecting to Bluetooth Speaker..."
bluetoothctl connect A2:75:E5:3B:61:2B # Replace with your speaker's MAC
sleep 2 # Wait for connection to stabilize

# ... (inside your start_system.sh script) ...

echo " -> Installing Edge AI Python Libraries..."
pip install --upgrade pip >> $LOG_FILE 2>&1

# This is the "Magic" line for Raspberry Pi compatibility
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite-runtime >> $LOG_FILE 2>&1

PIP_LIBS="opencv-python-headless streamlit psutil pandas matplotlib requests"
for lib in $PIP_LIBS; do
    echo "    Installing $lib..."
    if pip install $lib >> $LOG_FILE 2>&1; then
        echo "    [OK] $lib"
    else
        echo "    [FAIL] $lib - Check $LOG_FILE"
    fi
done

# Launch Streamlit (The Web Dashboard)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0