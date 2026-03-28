import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os

# --- CONFIGURATION ---
MODEL_PATH = "/home/edgeai/Desktop/SignLanguageEdge/Model_file/sign_mobilenet.tflite" # Path to your CNN model
THRESHOLD_T = 138.92
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

# --- 1. INITIALIZE AI BRAIN ---
print(f"?? Loading AI Brain: {MODEL_PATH}...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("? AI Loaded Successfully!")
except Exception as e:
    print(f"? Error loading model: {e}")
    exit()

# --- 2. PRE-PROCESSING (Mirroring your training logic) ---
def process_frame(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2000:
            cv2.drawContours(final_canvas, [c], -1, 255, -1) 
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

# --- 3. TEST LOOP ---
cap = cv2.VideoCapture(0)
print("?? Starting Live Inference Test... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Pre-process
    processed = process_frame(frame)
    
    # Run AI
    input_tensor = processed.reshape(1, 96, 96, 1)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get Result
    idx = np.argmax(output)
    confidence = output[idx]
    label = CLASSES[idx]

    # Show on terminal
    if confidence > 0.85:
        print(f"PREDICTION: {label:10} | CONFIDENCE: {confidence*100:3.1f}%", end='\r')
    else:
        print(f"PREDICTION: Waiting...   | CONFIDENCE: {confidence*100:3.1f}%", end='\r')

    # Visual Feedback (Small windows)
    # Binary mask shows you what the AI is 'looking' at
    cv2.imshow("Original Feed", frame)
    cv2.imshow("AI Input (96x96)", processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTest Stopped.")