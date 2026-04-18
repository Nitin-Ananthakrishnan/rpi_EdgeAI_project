import cv2
import numpy as np
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
        print("🚨 AI Engine Missing!")
        exit()

# --- 2. IMPORT CUSTOM MODULES ---
import services
from nlp_engine import NLPEngine
nlp = NLPEngine()

# --- 3. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_FOLDER = "Model_file" 
# Must match Colab exactly
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# CALCULATED PARAMETERS
THRESHOLD_T = 138.92
STABILITY_REQ = 8 # Lowered for more responsive demo
CONFIDENCE_GATE = 0.90
MIN_AREA = 2500

# --- 4. PRE-PROCESSING ---
def process_frame(frame):
    small = cv2.resize(frame, (320, 240))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > MIN_AREA:
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0, mask_temp
            
    return np.zeros((96, 96), dtype='float32'), final_canvas

# --- 5. INITIALIZE MODEL ---
ACTIVE_MODEL = "mobilenet" 
model_path = BASE_DIR / MODEL_FOLDER / f"sign_{ACTIVE_MODEL}.tflite"

try:
    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"❌ Load Error: {e}"); exit()

# --- 6. MAIN LOOP ---
cap = cv2.VideoCapture(0)
print(f"\n🚀 SYSTEM ONLINE | Engine: {ENGINE_NAME}")
print("--------------------------------------------------------------")

stable_label = ""
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        start_t = time.time()
        
        # A. PERCEPTION
        ai_input, debug_mask = process_frame(frame)
        
        # B. INFERENCE (Initialize variables FIRST to avoid NameError)
        label = "Background"
        conf = 1.0
        
        if np.max(ai_input) > 0:
            input_tensor = ai_input.reshape(1, 96, 96, 1)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            idx = np.argmax(output)
            label = CLASSES[idx]
            conf = output[idx]
        
        latency = (time.time() - start_t) * 1000
        
        # C. UI OVERLAY
        cv2.putText(frame, f"AI: {label} ({conf*100:.0f}%)", (10, 40), 2, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"{latency:.1f}ms", (10, 70), 2, 0.6, (255, 255, 255), 1)
        cv2.imshow("Live Feed", frame)
        cv2.imshow("AI Mask", debug_mask)

        # D. COGNITION & SPEECH (Memory persistence fix)
        if label != "Background" and conf > CONFIDENCE_GATE:
            if label == stable_label:
                frame_counter += 1
            else:
                stable_label, frame_counter = label, 0
                
            if frame_counter == STABILITY_REQ:
                # Calls nlp_engine which uses its own internal 5-second timer
                sentence = nlp.process_and_speak(label)
                if sentence:
                    print(f"\n[!] SPEECH: {sentence}")
                frame_counter = -30 # Cooldown
        
        # We NO LONGER clear NLP memory here if label is Background.
        # The NLP Engine now manages its own 5-second context window.

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()
