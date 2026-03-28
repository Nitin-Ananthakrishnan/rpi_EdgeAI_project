import cv2
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
import time
import psutil
import os
import datetime
import services  # Your API service file
from nlp_engine import NLPEngine  # Your NLP logic file

# --- 1. SYSTEM CONFIGURATION & PARAMETERS ---
# Calculated 3-Sigma Threshold from our Statistical Analysis
THRESHOLD_T = 138.92 
# The exact order used during your Colab Training
# MUST match the exact order of your TARGET_GESTURES list from Colab
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# Initialize the NLP Engine
nlp = NLPEngine()

# Telemetry Log File
LOG_FILE = "system_telemetry.txt"

# --- 2. TELEMETRY LOGGER ---
def log_telemetry(model_name, latency_ms):
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    try:
        temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
    except:
        temp = "N/A"
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] Model: {model_name} | Latency: {latency_ms:.1f}ms | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}C\n"
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

# --- 3. PRE-PROCESSING PIPELINE (Cr-Mean Filter) ---
def process_frame(frame):
    # 1. Standardize size for processing
    frame_resized = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    # 2. Otsu Binarization (Calculated Parameter)
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Geometric Filtering
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Mean Cr Validation (The 3-Sigma Guard)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        # THE CALCULATED THRESHOLD: 138.92
        if mean_cr > 138.92 and cv2.contourArea(c) > 2000:
            # DRAW FILLED CONTOUR (Matches Colab training perfectly)
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            
            # Crop to the gesture
            x, y, w, h = cv2.boundingRect(c)
            pad = 15
            crop = final_canvas[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
            
            # Resize and Normalize for AI input
            resized = cv2.resize(crop, (96, 96))
            return resized.astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

# --- 4. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Edge AI Assistive Hub", layout="wide")
st.title("🤖 Autonomous Edge AI Sign Language System")

# Sidebar
st.sidebar.header("System Controls")
model_choice = st.sidebar.selectbox("Active Algorithm", ["mobilenet", "lstm", "gru"])
run_system = st.sidebar.checkbox("Start System", value=True)

# Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Visual Perception")
    frame_placeholder = st.empty()
with col2:
    st.header("Cognitive Output")
    prediction_display = st.empty()
    st.divider()
    st.header("Hardware Telemetry")
    telemetry_display = st.empty()

# --- 5. LOAD AI MODEL ---
model_path = f"sign_{model_choice}.tflite"
try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success(f"Brain Loaded: {model_choice}")
except:
    st.sidebar.warning("Waiting for model files...")
    st.stop()

# --- 6. MAIN EXECUTION LOOP ---
cap = cv2.VideoCapture(0)
stable_label = ""
frame_counter = 0
STABILITY_REQ = 15 # Frames required to confirm a sign (~0.5 seconds)

while run_system:
    ret, frame = cap.read()
    if not ret: break
    
    start_t = time.time()
    
    # Perception Phase
    processed = process_frame(frame)
    
    # Inference Phase
    if model_choice == "mobilenet":
        input_tensor = processed.reshape(1, 96, 96, 1)
    else:
        # Sequence models expect 5 frames
        input_tensor = np.repeat(processed[np.newaxis, :, :], 5, axis=0).reshape(1, 5, 96, 96, 1)
        
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Logic Phase
    idx = np.argmax(output)
    conf = output[idx]
    current_prediction = CLASSES[idx]
    
    # --- TEMPORAL DEBOUNCING (Stability Logic) ---
    if conf > 0.85:
        if current_prediction == stable_label:
            frame_counter += 1
        else:
            stable_label = current_prediction
            frame_counter = 0
            
        if frame_counter == STABILITY_REQ:
            if stable_label != "Background":
                # Cognition Phase (NLP + Audio)
                # This function in nlp_engine handles the speech and API calls
                nlp.process_and_speak(stable_label) 
                prediction_display.success(f"INTENT: {stable_label}")
            else:
                nlp.previous_sign = None # Reset memory on background
                prediction_display.info("Awaiting Gesture...")
    
    # Telemetry and UI Update
    latency = (time.time() - start_t) * 1000
    log_telemetry(model_choice, latency)
    
    frame_placeholder.image(frame, channels="BGR")
    telemetry_display.code(f"""
    Model:      {model_choice.upper()}
    Latency:    {latency:.1f} ms
    FPS:        {1000/latency:.1f}
    Confidence: {conf*100:.1f}%
    """)

cap.release()
