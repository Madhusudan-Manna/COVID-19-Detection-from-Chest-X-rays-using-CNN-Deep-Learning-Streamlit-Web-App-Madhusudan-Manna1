import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("best_model.keras")

classes = ['COVID', 'NORMAL', 'PNEUMONIA']

st.title("Chest X-ray Classification")

file = st.file_uploader("Upload X-ray", type=['jpg','png','jpeg'])

if file:

    img = Image.open(file).resize((128,128))

    st.image(img, use_container_width=True)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    result = classes[np.argmax(pred)]
    conf = np.max(pred) * 100

    st.success(f"{result} ({conf:.2f}%)")