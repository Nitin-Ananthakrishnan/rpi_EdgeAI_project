import cv2
import numpy as np
import time
import os
from pathlib import Path
from tabulate import tabulate

# --- 1. UNIVERSAL AI ENGINE IMPORT ---
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- 2. CONFIGURATION ---
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "Model_file" / "sign_mobilenet.tflite"
CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']
THRESHOLD_T = 138.92 

# --- 3. OPTIMIZED PRE-PROCESSING (Returns Image + Aspect Ratio) ---
def process_frame(frame):
    small = cv2.resize(frame, (640, 480))
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_canvas = np.zeros_like(mask)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask_t = np.zeros_like(mask)
        cv2.drawContours(mask_t, [c], -1, 255, -1)
        
        if cv2.mean(cr, mask=mask_t)[0] > 135 and cv2.contourArea(c) > 1000:
            # --- THE CALCULATED PARAMETER: ASPECT RATIO ---
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h 
            
            cv2.drawContours(final_canvas, [c], -1, 255, -1)
            crop = final_canvas[y:y+h, x:x+w]
            resized = cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            return resized, aspect_ratio
            
    return np.zeros((96, 96), dtype='float32'), 0.0

# --- 4. INITIALIZE ---
interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
test_results = []

print("\n" + "="*55)
print("     EDGE AI SYSTEM: LIVE HARDWARE VALIDATION")
print("="*55)

# --- 5. TEST LOOP ---
try:
    for target in CLASSES:
        if target == "Background": continue
        
        print(f"👉 Prepare gesture for [{target.upper()}]")
        while True:
            ret, frame = cap.read()
            display_frame = frame.copy()
            cv2.rectangle(display_frame, (10, 10), (630, 90), (0,0,0), -1)
            cv2.putText(display_frame, f"TESTING: {target.upper()}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "READY? Press [SPACE] to start countdown", (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Validation Hub", display_frame)
            if cv2.waitKey(1) & 0xFF == 32: break

        for i in range(3, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = cap.read()
                cv2.putText(frame, str(i), (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                cv2.imshow("Validation Hub", frame)
                cv2.waitKey(1)

        ret, frame = cap.read()
        
        # --- FIXED: UNPACKING BOTH VALUES ---
        ai_input, aspect_ratio = process_frame(frame)
        
        input_tensor = ai_input.reshape(1, 96, 96, 1)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        idx = np.argmax(output)
        label = CLASSES[idx]
        conf = output[idx]

        # --- 6. HEURISTIC REFINEMENT LAYER (The Mark Winner) ---
        if label == "Raised" and aspect_ratio > 0.85:
            label = "Thumbsup"
        elif label == "L" and aspect_ratio > 1.2:
            label = "Call"
        elif label == "Pinch" and aspect_ratio < 0.6:
            label = "Pointing"
            
        pred_label = label
        status = "✅ PASS" if pred_label.lower() == target.lower() else "❌ FAIL"
        test_results.append([target, pred_label, f"{conf*100:.1f}%", status])
        
        color = (0, 255, 0) if status == "✅ PASS" else (0, 0, 255)
        res_img = frame.copy()
        cv2.rectangle(res_img, (10, 350), (630, 470), (0,0,0), -1)
        cv2.putText(res_img, f"AI + MATH SAID: {pred_label}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(res_img, f"RATIO: {aspect_ratio:.2f}", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Validation Hub", res_img)
        cv2.waitKey(2000)

    print("\n" + "="*60)
    print("           FINAL HARDWARE ACCURACY REPORT")
    print("="*60)
    print(tabulate(test_results, headers=["TARGET", "PREDICTION", "CONF", "STATUS"], tablefmt="grid"))
    total_pass = sum(1 for r in test_results if r[3] == "✅ PASS")
    print(f"\n🎯 ON-DEVICE RELIABILITY: {(total_pass / len(test_results)) * 100:.1f}%")

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    cap.release()
    cv2.destroyAllWindows()
