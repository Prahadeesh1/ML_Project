# garlic_vs_pastry_app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# === Load model ===
@st.cache_resource
def load_trained_model():
    return load_model(
        r'C:\Users\praha\Downloads\AMLAI_PROJECT\AMLAI_PROJECT\best_smooth_modelv14.keras',
        compile=False
    )

model = load_trained_model()

# === Class labels (⚠️ adjust if your label order is different!) ===
class_labels = {0:'bread_pastry' , 1:'garlic' , 2: 'unknown'}

# === App title ===
st.title("🧄 Garlic vs 🥐 Bread/Pastry Classifier")
st.write("Upload an image and the model will classify it into garlic, bread/pastry, or unknown.")

# === File uploader ===
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((180, 180))  # Change if your input shape is different
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]  # Output shape: (3,)
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # === Results ===
    st.markdown(f"### 🔎 Prediction: **{class_labels[predicted_class]}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # === Show all class probabilities ===
    st.subheader("📊 Class Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"**{class_labels[i]}**: {prob * 100:.2f}%")
