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

# ?? THE FIX: EXACT CUSTOM COLAB ORDER (NOT ALPHABETICAL) ??
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
MODELS = ["mobilenet", "lstm", "gru"]

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

def run_benchmark(model_name):
    model_path = MODEL_DIR / f"sign_{model_name}.tflite"
    if not model_path.exists():
        return [model_name.upper(), "FILE ERROR", "-", "-", "-", "-"]

    interpreter = tflite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct, total = 0, 0
    latencies = []
    
    print(f"\n--- Benchmarking {model_name.upper()} ---")
    files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    
    for img_name in files:
        # e.g., "Hello_3.jpg" -> "hello"
        actual_label = img_name.split('_')[0].lower()
        
        # Load the ALREADY PROCESSED 96x96 mask
        img = cv2.imread(str(TEST_IMG_DIR / img_name), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Clean JPEG Artifacts & Normalize
        img_resized = cv2.resize(img, (96, 96))
        _, img_binary = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
        img_input = img_binary.astype('float32') / 255.0 

        # Prepare Tensor
        if model_name == "mobilenet":
            input_tensor = img_input.reshape(1, 96, 96, 1)
        else:
            input_tensor = np.repeat(img_input[np.newaxis,:,:], 5, axis=0).reshape(1, 5, 96, 96, 1)

        # Inference
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        latencies.append((time.time() - start) * 1000)

        # Analysis
        idx = np.argmax(output)
        pred_label = CLASSES[idx].lower()
        
        if pred_label == actual_label:
            correct += 1
            
        total += 1

    acc = (correct / total) * 100 if total > 0 else 0
    avg_lat = np.mean(latencies) if latencies else 0
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    temp = os.popen("vcgencmd measure_temp").readline().replace("temp=","").strip()
    
    return [model_name.upper(), f"{acc:.1f}%", f"{avg_lat:.1f}ms", f"{cpu}%", f"{ram}%", temp]

# --- EXECUTION ---
print("?? Launching Final Benchmark with Custom Mapping...")
results = [run_benchmark(m) for m in MODELS]

headers = ["ALGORITHM", "TRUE ACCURACY", "AVG LATENCY", "CPU LOAD", "RAM USAGE", "TEMP"]
print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
