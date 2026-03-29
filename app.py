import cv2
import numpy as np
import streamlit as st
import time
import psutil
import os
import datetime
from pathlib import Path

# --- 1. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
    ENGINE_NAME = "LiteRT (Native 3.13)"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        ENGINE_NAME = "TFLite-Runtime"
    except ImportError:
        st.error("🚨 AI Engine Missing!")
        st.stop()

import services
from nlp_engine import NLPEngine
nlp = NLPEngine()

# --- 2. CALCULATED PARAMETERS & CONFIG ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
LOG_FILE = BASE_DIR / "system_telemetry.txt"
# 🔥 EXACT COLAB TRAINING ORDER 🔥
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92  # Derived from 3-Sigma Profiling
CONFIDENCE_LIMIT = 0.85 # Calculated Bayesian Gate (Rounded for stability)

# --- 3. TELEMETRY LOGGER ---
def log_telemetry(model_name, latency_ms):
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    try:
        temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","")
    except: temp = "N/A"
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {model_name} | Lat: {latency_ms:.1f}ms | CPU: {cpu}% | Temp: {temp}C\n")

# --- 4. PRE-PROCESSING (Cr-Mean Morphological Filter) ---
def process_frame(frame):
    # Resize immediately for 4x speed increase
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_t = np.zeros_like(mask); cv2.drawContours(mask_t, [c], -1, 255, -1)
        # Validation
        if cv2.mean(cr, mask=mask_t)[0] > THRESHOLD_T and cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            crop = mask_t[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask_t
    return np.zeros((96, 96), dtype='float32'), mask

# --- 5. UI SETUP ---
st.set_page_config(page_title="Edge AI Assistive Hub", layout="wide")
st.title("🤖 Autonomous Edge AI Sign Assistant")
model_choice = st.sidebar.selectbox("Active Algorithm", ["mobilenet", "lstm", "gru"])
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{model_choice}.tflite"

# Load Model
try:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success(f"✅ Loaded: {model_choice}")
except:
    st.sidebar.error("Model File Not Found!"); st.stop()

col1, col2 = st.columns([2, 1])
with col1: frame_placeholder = st.empty()
with col2: 
    prediction_display = st.empty()
    telemetry_display = st.empty()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)
stable_label, frame_counter = "", 0

while True:
    ret, frame = cap.read()
    if not ret: break
    start_t = time.time()
    
    # Perception
    ai_input, debug_mask = process_frame(frame)
    
    # Inference
    if model_choice == "mobilenet":
        tensor = ai_input.reshape(1, 96, 96, 1)
    else:
        tensor = np.repeat(ai_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)
        
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Decision Logic
    idx = np.argmax(output)
    conf = output[idx]
    label = CLASSES[idx]
    
    if conf > CONFIDENCE_LIMIT:
        if label == stable_label: frame_counter += 1
        else: stable_label, frame_counter = label, 0
            
        if frame_counter == 15: # 15 Frames = ~0.5s stability
            if label != "Background":
                sentence = nlp.process_and_speak(label)
                prediction_display.success(f"INTENT: {label}\n\nSPEECH: {sentence}")
            else:
                nlp.previous_sign = None
                prediction_display.info("Awaiting Gesture...")

    latency = (time.time() - start_t) * 1000
    log_telemetry(model_choice, latency)
    frame_placeholder.image(frame, channels="BGR", caption=f"Processing at {1000/latency:.1f} FPS")
    telemetry_display.code(f"Model: {model_choice.upper()}\nLatency: {latency:.1f}ms\nCPU: {psutil.cpu_percent()}%")
