import cv2
import numpy as np
import time
import psutil
import os
import datetime
from pathlib import Path
from tabulate import tabulate

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

# --- 2. CONFIGURATION (Path Corrected) ---
# This script is in ~/Desktop/SignLanguageEdge/Hardware_testing/
CURRENT_DIR = Path(__file__).resolve().parent

# pi_test_set is in the same folder as this script
TEST_IMG_DIR = CURRENT_DIR / "pi_test_set"

# Model_file is one level up
MODEL_DIR = CURRENT_DIR.parent / "Model_file"

# THE TRUTH MAP: Exact order from Colab training
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92 

# --- 3. PRE-PROCESSING PIPELINE (The "Eyes" of the system) ---
def process_frame(frame):
    # Standardize to 480p for internal math
    small = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    
    # Otsu Binarization
    blur = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_temp = np.zeros_like(mask)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)
        mean_cr = cv2.mean(cr, mask=mask_temp)[0]
        
        # 3-Sigma and Area Rejection
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2000:
            # Filled Contour (Preserves finger gaps)
            cv2.drawContours(final_canvas, [c], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[y:y+h, x:x+w]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

# --- 4. BENCHMARKING ENGINE ---
def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), f"MISSING FILE: {model_path.name}", "-", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    print(f"\n🚀 Testing Algorithm: {model_name.upper()}")
    
    # Sort files to ensure deterministic results
    files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))])
    
    if not files:
        return [model_name.upper(), "NO IMAGES FOUND", "-", "-", "-", "-"]

    for img_name in files:
        # Expected filename format: "Hello_5.jpg" -> class "hello"
        actual_label = img_name.split('_')[0].lower()
        
        raw_img = cv2.imread(str(TEST_IMG_DIR / img_name))
        if raw_img is None: continue
        
        # Perception
        ai_input = process_frame(raw_img)
        
        # Tensor Preparation
        if model_name == "mobilenet":
            input_tensor = ai_input.reshape(1, 96, 96, 1)
        else:
            input_tensor = np.repeat(ai_input[np.newaxis, :, :], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # Inference
        start_t = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start_t) * 1000)

        # Accuracy
        idx = np.argmax(output)
        if CLASSES[idx].lower() == actual_label:
            correct += 1
        
        total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies)
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").strip()

    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.1f}ms", f"{cpu}%", f"{ram}%", temp]

# --- 5. EXECUTION ---
print("================================================================")
print(f"📡 HARDWARE EVALUATION SUITE | Engine: {ENGINE_NAME}")
print(f"📁 Test Folder: {TEST_IMG_DIR}")
print("================================================================")

results = [run_benchmark(m) for m in ["mobilenet", "lstm", "gru"]]

headers = ["ALGORITHM", "ACCURACY", "LATENCY", "CPU", "RAM", "TEMP"]
print("\n" + tabulate(results, headers=headers, tablefmt="grid"))

with open("hardware_final_report.txt", "w") as f:
    f.write(f"=== SYSTEM COMPARATIVE ANALYSIS: {datetime.datetime.now()} ===\n")
    f.write(tabulate(results, headers=headers, tablefmt="grid"))
