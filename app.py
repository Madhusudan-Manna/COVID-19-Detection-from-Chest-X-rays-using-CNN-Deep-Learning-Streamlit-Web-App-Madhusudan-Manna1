import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("best_model.keras")

classes = ['COVID', 'NORMAL', 'PNEUMONIA']

st.title("COVID-19 Detection from Chest X-rays")

file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "png", "jpeg"]
)

if file:

    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    # Preprocess
    img = img.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)

    result = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")