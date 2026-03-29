import cv2
import numpy as np
import time
import psutil
import os
from pathlib import Path
from tabulate import tabulate

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = CURRENT_SCRIPT_DIR / "pi_test_set"
MODEL_DIR = CURRENT_SCRIPT_DIR.parent / "Model_file"

CLASSES = ['Background', 'Call', 'Hello', 'L', 'Peace', 'Pinch', 'Pointing', 'Raised', 'Thumbsup', 'Yes']
MODELS = ["mobilenet", "lstm", "gru"]
THRESHOLD_T = 138.92

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

def process_frame(frame):
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
        
        if mean_cr > THRESHOLD_T and cv2.contourArea(c) > 2500:
            hull = cv2.convexHull(c)
            cv2.drawContours(final_canvas, [hull], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(c)
            crop = final_canvas[max(0,y-10):y+h+10, max(0,x-10):x+w+10]
            return cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
    return np.zeros((96, 96), dtype='float32')

def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), "FILE ERROR", "-", "-", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total, rejected_count = 0, 0, 0
    latencies = []
    
    print(f"\n--- Benchmarking {model_name.upper()} ---")
    files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    for img_name in files:
        actual_label = img_name.split('_')[0].lower()
        
        raw_img = cv2.imread(str(TEST_IMG_DIR / img_name))
        if raw_img is None: continue
        
        ai_input = process_frame(raw_img)
        
        # THE DIAGNOSTIC CHECK: Is the image solid black?
        if np.max(ai_input) == 0 and actual_label != "background":
            rejected_count += 1
            # We skip scoring this image because the pre-processor failed, not the AI.
            continue

        if model_name == "mobilenet":
            input_tensor = ai_input.reshape(1, 96, 96, 1)
        else:
            input_tensor = np.repeat(ai_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start) * 1000)

        idx = np.argmax(output)
        pred_label = CLASSES[idx].lower()
        
        if pred_label == actual_label:
            correct += 1
            
        total += 1

    # We calculate accuracy based ONLY on images that the AI actually saw (not black ones)
    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies) if latencies else 0
    
    return [model_name.upper(), f"{acc:.1f}%", f"{rejected_count}/{len(files)}", f"{avg_lat:.1f}ms", f"{psutil.cpu_percent()}%", f"{psutil.virtual_memory().percent}%", os.popen("vcgencmd measure_temp").readline().replace("temp=","").strip()]

# --- EXECUTION ---
print("🚀 Launching Diagnostic Benchmark (Filtering Black Images)...")
results = [run_benchmark(m) for m in MODELS]

headers = ["ALGORITHM", "TRUE ACCURACY", "REJECTED BY FILTER", "LATENCY", "CPU", "RAM", "TEMP"]
print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
