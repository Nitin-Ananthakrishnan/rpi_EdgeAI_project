#!/bin/bash

# --- 1. SET UP THE ENVIRONMENT ---
echo "[1/4] Activating Virtual Environment..."
cd ~/SignLanguageEdge
source venv/bin/activate

# --- 2. CONFIGURE HARDWARE ---
echo "[2/4] Configuring Audio & Video Peripherals..."
# Force audio to USB Speaker (replace '1' with your speaker card ID if needed)
amixer cset numid=3 1 > /dev/null 2>&1 

# Set permissions for Camera and Audio
sudo usermod -a -G video $USER
sudo usermod -a -G audio $USER

# --- 3. TELEMETRY AUDIT ---
echo "[3/4] Initializing System Telemetry..."
echo "Timestamp,Model,CPU,RAM,Temp,Latency" > system_telemetry.csv

# --- 4. LAUNCH ASSISTIVE SYSTEM ---
echo "[4/4] Launching Streamlit Dashboard..."
# We use nohup so the system keeps running even if the terminal disconnects
# We map to 0.0.0.0 so you can access it from your phone/laptop browser
streamlit run app.py --server.port 8501 --server.address 0.0.0.0