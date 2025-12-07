from PIL import Image, ImageOps
import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import platform

# === Environment Info ===
print("=" * 40)
print("📦 Environment Info")
print(f"🐍 Python version      : {platform.python_version()}")
print(f"🔢 TensorFlow version  : {tf.__version__}")
print("=" * 40)

# === Class Mapping (ensure correct order!) ===
class_labels = {0: 'garlic', 1: 'bread_pastry' , 2: 'unknown'}

# === Prediction Function ===
def import_and_predict(image_data, model):
    size = (185, 185)  # Match training input size
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction[0]

# === Load Model ===
model = tf.keras.models.load_model(
    r"C:\Users\praha\Downloads\AMLAI_PROJECT\AMLAI_PROJECT\best_smooth_modelv15.keras"
)

# === Open Camera ===
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("📷 Camera OK")
else:
    print("❌ Failed to open camera.")
    sys.exit()

# === Threshold ===
CONFIDENCE_THRESHOLD = 0.60

while True:
    ret, original = cap.read()
    if not ret or original is None:
        print("❌ Failed to grab frame.")
        continue

    display_frame = original.copy()
    cv2.imwrite('img.jpg', original)
    image = Image.open('img.jpg')

    # Predict
    pred_vector = import_and_predict(image, model)
    predicted_class = np.argmax(pred_vector)
    confidence = pred_vector[predicted_class]
    class_name = class_labels[predicted_class]

    print(f"📊 Prediction: {class_name} ({confidence*100:.2f}%)")
    print(f"→ Raw scores: {pred_vector}")

    # === Main Label Display ===
    if confidence >= CONFIDENCE_THRESHOLD:
        label_text = f"{class_name.capitalize()} ({confidence * 100:.2f}%)"
        text_color = (0, 255, 0)  # green
    else:
        label_text = f"Uncertain: {class_name} ({confidence * 100:.2f}%)"
        text_color = (0, 0, 255)  # red

    # Draw main prediction
    cv2.putText(display_frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # === Show all class probabilities ===
    for i, prob in enumerate(pred_vector):
        label = f"{class_labels[i]}: {prob * 100:.2f}%"
        y = 65 + i * 30

        # Draw black background rectangle
        cv2.rectangle(display_frame, (10, y - 20), (320, y + 5), (0, 0, 0), -1)

        # Draw text: green if above threshold, else white
        color = (0, 255, 0) if prob >= CONFIDENCE_THRESHOLD else (255, 255, 255)
        cv2.putText(display_frame, label, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show live video
    cv2.imshow("Classification", display_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
