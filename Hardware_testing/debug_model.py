import cv2
import numpy as np
import os
from pathlib import Path

# --- 1. CONFIGURATION ---
CURRENT_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = CURRENT_DIR / "pi_test_set"
MODEL_PATH = CURRENT_DIR.parent / "Model_file" / "sign_mobilenet.tflite"

CLASSES = ['Background', 'Hello', 'Yes', 'Thumbsup', 'Pointing', 'Raised', 'Pinch', 'Call', 'Peace', 'L']

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

print("\n" + "="*50)
print("?? NEURAL NETWORK X-RAY DIAGNOSTIC")
print("="*50)

# --- 2. FIND A TEST IMAGE ---
test_image_path = None
for f in os.listdir(TEST_IMG_DIR):
    if f.lower().startswith("hello"): # Let's test a "Hello" sign
        test_image_path = TEST_IMG_DIR / f
        break

if test_image_path is None:
    print("? Could not find a 'Hello' image in pi_test_set.")
    exit()

print(f"?? Testing Image: {test_image_path.name}")

# --- 3. LOAD AND FORMAT IMAGE EXACTLY LIKE TRAINING ---
img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (96, 96))

# CRITICAL FIX: JPEG artifacts create gray pixels. We FORCE it back to pure 0 and 1.
_, img_binary = cv2.threshold(img_resized, 127, 255, cv2.THRESH_BINARY)
img_input = img_binary.astype('float32') / 255.0

# --- 4. RUN INFERENCE ---
interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(in_idx, img_input.reshape(1, 96, 96, 1))
interpreter.invoke()
output_array = interpreter.get_tensor(out_idx)[0]

# --- 5. THE DIAGNOSIS ---
print("\n?? RAW PROBABILITY OUTPUT:")
for i, prob in enumerate(output_array):
    print(f"Index {i:2d} ({CLASSES[i]:10}): {prob*100:6.2f}%")

print("\n" + "="*50)
max_idx = np.argmax(output_array)
print(f"?? AI PREDICTED: Index {max_idx} with {output_array[max_idx]*100:.2f}% confidence.")
print("="*50)