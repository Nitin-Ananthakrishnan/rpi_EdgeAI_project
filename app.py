import cv2
import numpy as np
import streamlit as st  # <--- CRITICAL IMPORT
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
        st.error("🚨 AI Engine Missing! Please ensure 'ai-edge-litert' is installed.")
        st.stop()

# --- 2. IMPORT CUSTOM MODULES ---
# These must exist as nlp_engine.py and services.py in the same folder
try:
    import services
    from nlp_engine import NLPEngine
    nlp = NLPEngine()
except ImportError as e:
    st.error(f"🚨 Module Missing: {e}")
    st.stop()

# --- 3. PATH & DATA CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
LOG_FILE = BASE_DIR / "system_telemetry.txt"
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92

# --- 4. TELEMETRY LOGGER ---
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

# --- 5. PRE-PROCESSING FUNCTION (Cr-Mean Filter) ---
def process_frame(frame):
    # Standardize size for processing
    frame_resized = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    # Otsu Binarization
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Contour Analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        # Validation
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2500:
            # Draw solid filled contour (Preserves finger gaps)
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            pad = 15
            crop = final_canvas[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

# --- 6. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Edge AI Assistive Hub", layout="wide")
st.title("🤖 Autonomous Edge AI Sign Language System")

st.sidebar.header("System Controls")
model_choice = st.sidebar.selectbox("Active Algorithm", ["mobilenet", "lstm", "gru"])
run_system = st.sidebar.checkbox("Start Live System", value=True)

# --- 7. LOAD THE SELECTED MODEL ---
model_filename = f"sign_{model_choice}.tflite"
model_path = BASE_DIR / MODEL_FOLDER / model_filename

try:
    # Use str() to ensure the path is a string for the interpreter
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success(f"✅ {ENGINE_NAME}\nModel: {model_filename}")
except Exception as e:
    st.sidebar.error(f"❌ Load Error: {e}")
    st.sidebar.write(f"Path tried: {model_path}")
    st.stop()

# --- 8. INITIALIZE HARDWARE (The fix for your NameError) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("🚨 Camera Error: System could not detect USB Vision Sensor.")
    st.stop()

# Dashboard Placeholders
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

# Temporal Variables
stable_label = ""
frame_counter = 0
STABILITY_REQ = 15 

# --- 9. MAIN INFERENCE LOOP ---
while run_system:
    ret, frame = cap.read()
    if not ret: break
    
    start_t = time.time()
    
    # Run perception
    processed_input = process_frame(frame)
    
    # Run Inference
    if model_choice == "mobilenet":
        # Shape: (1, 96, 96, 1)
        input_tensor = processed_input.reshape(1, 96, 96, 1)
    else:
        # Sequence Shape: (1, 5, 96, 96, 1)
        input_tensor = np.repeat(processed_input[np.newaxis, :, :], 5, axis=0).reshape(1, 5, 96, 96, 1)
        
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Process confidence
    idx = np.argmax(output)
    conf = output[idx]
    current_prediction = CLASSES[idx]
    
    # Stability Logic (Debouncing)
    if conf > 0.85:
        if current_prediction == stable_label:
            frame_counter += 1
        else:
            stable_label = current_prediction
            frame_counter = 0
            
        if frame_counter == STABILITY_REQ:
            if stable_label != "Background":
                # Cognition Phase
                sentence = nlp.process_and_speak(stable_label)
                if sentence:
                    prediction_display.success(f"GESTURE: {stable_label}\n\nSPEECH: {sentence}")
            else:
                nlp.previous_sign = None # Clear memory on Background
                prediction_display.info("System Ready. Awaiting Gesture...")

    # Performance Monitoring
    latency = (time.time() - start_t) * 1000
    log_telemetry(model_choice, latency)
    
    # Update UI
    frame_placeholder.image(frame, channels="BGR")
    telemetry_display.code(f"""
    Algorithm:  {model_choice.upper()}
    Latency:    {latency:.1f} ms
    FPS:        {1000/latency:.1f}
    Engine:     {ENGINE_NAME}
    Confidence: {conf*100:.1f}%
    """)

# Cleanup
cap.release()
