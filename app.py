import cv2
import numpy as np
import streamlit as st
import tflite_runtime.interpreter as tflite
import time
import psutil
import os
import datetime

# --- 1. TELEMETRY LOGGING (Text File) ---
LOG_FILE = "system_telemetry.txt"

def log_telemetry(model_name, latency_ms):
    # Get Hardware Metrics
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    
    # Raspberry Pi specific temperature command
    try:
        temp_str = os.popen("vcgencmd measure_temp").readline()
        temp = temp_str.replace("temp=", "").replace("'C\n", "")
    except:
        temp = "N/A" # Fallback if run on a non-Pi system
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the log entry as a readable text line
    log_entry = f"[{timestamp}] Model: {model_name} | Latency: {latency_ms:.1f}ms | CPU: {cpu}% | RAM: {ram}% | Temp: {temp}°C\n"
    
    # Append to the text file
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)

# --- 2. NLP ENGINE (Semantic Dictionary) ---
semantic_dictionary = {
    "HELLO": "Hello! I am using an AI assistant.",
    "YES": "Yes, I agree.",
    "HELP": "I need emergency assistance please.",
    "THANKYOU": "Thank you very much.",
    "POINT": "I would like that item, please.",
    "THUMBSUP": "I am doing good, everything is okay."
}

# --- 3. PRE-PROCESSING (The "Cr-Mean Filter") ---
# The Calculated Threshold for Skin-Tone (from our Statistical Analysis)
THRESHOLD_T = 138.92 

def process_frame(frame):
    # Resize to 640x480 to save CPU cycles before math
    frame = cv2.resize(frame, (640, 480))
    
    # YCrCb Conversion
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    # Otsu's Binarization on the Red-Difference channel
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Mean Cr Validation (Is it actually skin?)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        # Spatial Validation (Is it large enough?)
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2000:
            # Solidify the hand shape
            hull = cv2.convexHull(c)
            cv2.drawContours(final_canvas, [hull], -1, 255, -1)
            
            # Crop tightly around the hand
            x, y, w, h = cv2.boundingRect(c)
            pad = 15
            crop = final_canvas[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]
            
            # Resize for the AI Model and Normalize [0, 1]
            resized = cv2.resize(crop, (96, 96))
            return resized.astype('float32') / 255.0
            
    # If no hand found, return the Null Background
    return np.zeros((96, 96), dtype='float32')

# --- 4. STREAMLIT DASHBOARD UI ---
st.set_page_config(page_title="Edge AI Assistive System", layout="wide")
st.title("🚀 Autonomous Edge AI: Sign Language Telemetry")

st.sidebar.header("System Configuration")
# Select the algorithm you want to demonstrate
model_choice = st.sidebar.selectbox("Select Algorithm", ["mobilenet", "lstm", "gru"])
run_system = st.sidebar.checkbox("Start Live Demo", value=True)

# UI Layout Columns
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Feed")
    frame_placeholder = st.empty()
with col2:
    st.header("Assistive Output")
    prediction_text = st.empty()
    sentence_text = st.empty()
    st.header("System Metrics")
    telemetry_text = st.empty()

# --- 5. INITIALIZE THE SELECTED MODEL ---
# Ensure your model files are named sign_mobilenet.tflite, sign_lstm.tflite, etc.
model_path = f"sign_{model_choice}.tflite"

try:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.sidebar.success(f"Loaded: {model_choice}.tflite")
except Exception as e:
    st.sidebar.error(f"Model File Not Found: {model_path}")
    st.stop()

# --- 6. MAIN INFERENCE LOOP ---
# 0 is usually the Pi Camera or USB Camera
cap = cv2.VideoCapture(0) 

# Variables for Audio Cooldown
last_spoken_time = 0
AUDIO_COOLDOWN = 3.0 # Wait 3 seconds before speaking the same thing again
last_label = ""

while run_system:
    ret, frame = cap.read()
    if not ret: 
        st.error("Camera Error")
        break
    
    # 1. Start Timing
    start_t = time.time()
    
    # 2. Process Image
    processed_input = process_frame(frame)
    
    # 3. Format Input for the specific model architecture
    if model_choice == "mobilenet":
        # CNN expects (1, 96, 96, 1)
        input_tensor = np.expand_dims(processed_input, axis=0).reshape(1, 96, 96, 1)
    else:
        # LSTM/GRU expects sequence of 5 frames (1, 5, 96, 96, 1)
        input_tensor = np.repeat(processed_input[np.newaxis, :, :], 5, axis=0).reshape(1, 5, 96, 96, 1)

    # 4. Run AI Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # 5. Calculate Metrics
    latency = (time.time() - start_t) * 1000
    log_telemetry(model_choice, latency)
    
    # 6. Parse AI Output
    idx = np.argmax(output)
    confidence = output[idx]
    
    # MUST MATCH THE ORDER YOU USED IN COLAB TRAINING
    CLASSES = ['Background', 'HELLO', 'YES', 'HELP', 'THANKYOU', 'POINT', 'THUMBSUP'] 
    
    # 7. Assistive Logic & Output
    if confidence > 0.85: # 85% Confidence Threshold
        label = CLASSES[idx]
        
        if label != 'Background':
            sentence = semantic_dictionary.get(label, "Unrecognized Sign")
            
            # Update UI
            prediction_text.metric("Sign Detected", f"{label} ({confidence*100:.1f}%)")
            sentence_text.info(sentence)
            
            # Audio Output (Asynchronous Flite command)
            if label != last_label or (time.time() - last_spoken_time > AUDIO_COOLDOWN):
                # The '&' runs it in the background so the camera doesn't freeze
                os.system(f'flite -t "{sentence}" &') 
                last_spoken_time = time.time()
                last_label = label
        else:
            prediction_text.metric("Status", "Awaiting Input...")
            sentence_text.empty()
            last_label = ""
            
    # Update UI Video and Telemetry
    frame_placeholder.image(frame, channels="BGR")
    telemetry_text.text(f"Latency: {latency:.1f}ms\nModel: {model_choice}")

cap.release()
