import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("Manna.keras")

# Class labels
classes = ['COVID', 'Normal', 'Viral Pneumonia']

st.title("COVID-19 Detection from Chest X-rays")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")