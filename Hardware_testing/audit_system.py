import cv2
import numpy as np
import os
import time
from pathlib import Path

# --- 1. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        print("?? AI Engine Missing! Run: pip install ai-edge-litert")
        exit()

# --- 2. CONFIGURATION ---
CURRENT_DIR = Path(__file__).resolve().parent
# Look for images in the folder next to this script
TEST_IMG_DIR = CURRENT_DIR / "pi_test_set"
# Look for model one level up in Model_file
MODEL_PATH = CURRENT_DIR.parent / "Model_file" / "sign_mobilenet.tflite"

# This is our current 'Guess' at the order. 
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92

# --- 3. PRE-PROCESSOR (Exact mirror of deployment) ---
def process_frame(frame):
    if frame is None: return np.zeros((96, 96), dtype='float32')
    frame_resized = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
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
            hull = cv2.convexHull(c)
            cv2.drawContours(final_canvas, [hull], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
    return np.zeros((96, 96), dtype='float32')

# --- 4. INITIALIZE AI ---
if not MODEL_PATH.exists():
    print(f"? Model not found at: {MODEL_PATH}")
    exit()

interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

# --- 5. AUDIT LOOP ---
print("\n" + "="*70)
print("?? EDGE AI SYSTEM AUDIT LOG")
print("="*70)

# Sort files so the output is easy to read
files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))])

if not files:
    print(f"? No images found in: {TEST_IMG_DIR}")
    exit()

for f in files:
    actual_label = f.split('_')[0]
    raw_img = cv2.imread(str(TEST_IMG_DIR / f))
    
    # 1. Check Pre-processor
    ai_input = process_frame(raw_img)
    has_signal = "VALID" if np.max(ai_input) > 0 else "EMPTY/BLACK"
    
    # 2. Run AI
    interpreter.set_tensor(in_idx, ai_input.reshape(1, 96, 96, 1))
    interpreter.invoke()
    output = interpreter.get_tensor(out_idx)[0]
    
    # 3. Analyze Prediction
    pred_idx = np.argmax(output)
    pred_label = CLASSES[pred_idx]
    conf = output[pred_idx] * 100
    
    status = "?" if actual_label.lower() == pred_label.lower() else "?"
    print(f"{status} File: {f:15} | Signal: {has_signal:11} | Pred: {pred_label:10} ({conf:5.1f}%)")

print("="*70)
print("AUDIT COMPLETE. PLEASE PASTE THIS OUTPUT.")