import cv2
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
import time
import psutil
import os
from nlp_engine import NLPEngine

# --- SMART TFLITE IMPORT ---
try:
    # Try the new Google "LiteRT" name first
    import ai_edge_litert.interpreter as tflite
    print("Using LiteRT Engine")
except ImportError:
    try:
        # Fallback to the old "tflite-runtime" name
        import tflite_runtime.interpreter as tflite
        print("Using TFLite-Runtime Engine")
    except ImportError:
        st.error("?? AI Engine Missing: Please run 'pip install ai-edge-litert'")
        st.stop()

# --- CONFIGURATION ---
THRESHOLD_T = 138.92 # Calculated from Colab
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
nlp = NLPEngine()

def log_telemetry(model_name, latency):
    with open("system_telemetry.txt", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} | {model_name} | Lat: {latency:.1f}ms | CPU: {psutil.cpu_percent()}% | Temp: {os.popen('vcgencmd measure_temp').readline().strip()}\n")

def process_frame(frame):
    ycrcb = cv2.cvtColor(cv2.resize(frame, (640,480)), cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:,:,1]
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_t = np.zeros_like(cr); cv2.drawContours(mask_t, [c], -1, 255, -1)
        if cv2.mean(cr, mask=mask_t)[0] > THRESHOLD_T and cv2.contourArea(c) > 2500:
            x, y, w, h = cv2.boundingRect(c)
            crop = cv2.resize(mask_t[y:y+h, x:x+w], (96,96))
            return crop.astype('float32') / 255.0
    return np.zeros((96,96), dtype='float32')

# --- UI ---
st.title("Autonomous Edge AI System")
model_choice = st.sidebar.selectbox("Algorithm", ["mobilenet", "lstm", "gru"])
interpreter = tflite.Interpreter(model_path=f"modelfile/sign_{model_choice}.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
info_placeholder = st.empty()

# --- LOOP ---
stable_label, frame_count = "", 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    start = time.time()
    processed = process_frame(frame)
    
    # Inference
    input_data = processed.reshape(1,96,96,1) if model_choice=="mobilenet" else np.repeat(processed[np.newaxis,:,:],5,axis=0).reshape(1,5,96,96,1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    idx = np.argmax(output)
    if output[idx] > 0.85:
        curr = CLASSES[idx]
        if curr == stable_label: frame_count += 1
        else: stable_label, frame_count = curr, 0
        
        if frame_count == 15:
            sentence = nlp.process_and_speak(stable_label)
            if sentence: info_placeholder.success(f"Output: {sentence}")
            
    latency = (time.time() - start) * 1000
    log_telemetry(model_choice, latency)
    frame_placeholder.image(frame, channels="BGR")
