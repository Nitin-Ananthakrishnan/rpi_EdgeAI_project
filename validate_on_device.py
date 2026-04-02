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

# --- 3. PRE-PROCESSING ---
def process_frame(frame):
    # ... (existing ycrcb and mask code) ...
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.mean(cr, mask=mask_t)[0] > 135 and cv2.contourArea(c) > 1000:
            # --- CALCULATE GEOMETRIC PARAMETERS ---
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h # <--- THE CALCULATED PARAMETER
            
            cv2.drawContours(final_canvas, [c], -1, 255, -1)
            crop = final_canvas[y:y+h, x:x+w]
            resized = cv2.resize(crop, (96, 96)).astype('float32') / 255.0
            
            # Return BOTH the image and the aspect ratio
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
        
        # --- NEW LIVE PREPARATION PHASE ---
        print(f"👉 Prepare gesture for [{target.upper()}]")
        while True:
            ret, frame = cap.read()
            display_frame = frame.copy()
            
            # Draw UI on screen
            cv2.rectangle(display_frame, (10, 10), (630, 90), (0,0,0), -1)
            cv2.putText(display_frame, f"TESTING: {target.upper()}", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "READY? Press [SPACE] to start countdown", (20, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Validation Hub", display_frame)
            
            # Check if SPACEBAR (ASCII 32) is pressed
            if cv2.waitKey(1) & 0xFF == 32:
                break

        # COUNTDOWN PHASE (Still Live)
        for i in range(3, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1.0: # Show number for 1 second
                ret, frame = cap.read()
                cv2.putText(frame, str(i), (280, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                cv2.imshow("Validation Hub", frame)
                cv2.waitKey(1)

        # CAPTURE & INFER
        ret, frame = cap.read()
        ai_input = process_frame(frame)
        input_tensor = ai_input.reshape(1, 96, 96, 1)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Inference Result
        idx = np.argmax(output)
        label = CLASSES[idx]
        conf = output[idx]

# --- HEURISTIC REFINEMENT LAYER ---
# If AI says "Raised" but the hand is wide, it's actually "Thumbsup"
        if label == "Raised" and aspect_ratio > 0.85:
                label = "Thumbsup"
    
# If AI says "L" but the hand is very wide, it's actually "Call"
        if label == "L" and aspect_ratio > 1.2:
            label = "Call"

# If AI says "Pinch" but the hand is very thin, it's actually "Pointing"
        if label == "Pinch" and aspect_ratio < 0.6:
            label = "Pointing"
            
        pred_label = label
        
        conf = output[idx] * 100
        status = "✅ PASS" if pred_label.lower() == target.lower() else "❌ FAIL"
        test_results.append([target, pred_label, f"{conf:.1f}%", status])
        
        # RESULT FEEDBACK
        color = (0, 255, 0) if status == "✅ PASS" else (0, 0, 255)
        res_img = frame.copy()
        cv2.rectangle(res_img, (10, 350), (630, 470), (0,0,0), -1)
        cv2.putText(res_img, f"AI SAID: {pred_label}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(res_img, f"CONFIDENCE: {conf:.1f}%", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Validation Hub", res_img)
        cv2.waitKey(2000) # Show result for 2 seconds

    # --- 6. FINAL REPORT ---
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
