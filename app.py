import cv2
import numpy as np
import streamlit as st
import time
import psutil
import os
import datetime
from pathlib import Path

# --- 1. SET PAGE CONFIG (MUST BE FIRST LINE) ---
st.set_page_config(page_title="Edge AI Assistive Hub", layout="wide")

# --- 2. DEFINE UI LAYOUT IMMEDIATELY (Prevents Blank Screen) ---
st.title("🤖 Autonomous Edge AI Sign Assistant")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Visual Perception")
    frame_placeholder = st.empty()
    status_text = st.empty()

with col2:
    st.header("Cognitive Output")
    prediction_display = st.empty()
    st.divider()
    st.header("Hardware Telemetry")
    telemetry_display = st.empty()

# --- 3. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
    ENGINE_NAME = "LiteRT (Native 3.13)"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        ENGINE_NAME = "TFLite-Runtime"
    except ImportError:
        st.error("🚨 AI Engine Missing! Run: pip install ai-edge-litert")
        st.stop()

# --- 4. IMPORT CUSTOM MODULES ---
try:
    import services
    from nlp_engine import NLPEngine
    nlp = NLPEngine()
except Exception as e:
    st.error(f"🚨 Logic Module Error: {e}")
    st.stop()

# --- 5. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
LOG_FILE = BASE_DIR / "system_telemetry.txt"
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92

# --- 6. PRE-PROCESSING FUNCTION ---
def process_frame(frame):
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask); cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 1000:
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask
    return np.zeros((96, 96), dtype='float32'), mask

# --- 7. SIDEBAR CONTROLS & MODEL LOADING ---
st.sidebar.header("System Controls")
model_choice = st.sidebar.selectbox("Active Algorithm", ["mobilenet", "lstm", "gru"])
run_system = st.sidebar.checkbox("🚀 Start System", value=False)

model_path = BASE_DIR / MODEL_FOLDER / f"sign_{model_choice}.tflite"

# Load the model only when selected
try:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success(f"✅ Brain Loaded: {model_choice}")
except Exception as e:
    st.sidebar.error(f"❌ Model Load Fail: {e}")
    st.stop()

# --- 8. MAIN INFERENCE LOOP ---
if run_system:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚨 Camera not found. Check USB connection.")
    else:
        stable_label, frame_counter = "", 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            start_t = time.time()
            ai_input, debug_mask = process_frame(frame)
            
            # Inference
            if model_choice == "mobilenet":
                tensor = ai_input.reshape(1, 96, 96, 1)
            else:
                tensor = np.repeat(ai_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)
                
            interpreter.set_tensor(input_details[0]['index'], tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            
            idx = np.argmax(output)
            conf = output[idx]
            label = CLASSES[idx]
            
            # NLP Logic
            if conf > 0.85:
                if label == stable_label: frame_counter += 1
                else: stable_label, frame_counter = label, 0
                
                if frame_counter == 10:
                    if label != "Background":
                        sentence = nlp.process_and_speak(label)
                        prediction_display.success(f"INTENT: {label}\n\nSPEECH: {sentence}")
                    else:
                        prediction_display.info("Awaiting Gesture...")
                        nlp.previous_sign = None

            # Performance Telemetry
            latency = (time.time() - start_t) * 1000
            frame_placeholder.image(frame, channels="BGR")
            telemetry_display.code(f"Latency: {latency:.1f}ms\nFPS: {1000/latency:.1f}\nCPU: {psutil.cpu_percent()}%")
            
    cap.release()
else:
    st.info("👈 Check the 'Start System' box in the sidebar to begin.")
