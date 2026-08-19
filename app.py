import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import base64
from io import BytesIO

# Model path
MODEL_PATH = "model.h5"

# Class names
class_names = ['No Tumor', 'Tumor']

# Page configuration
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Title and description
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI image to predict if a tumor is present.")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if uploaded_file is not None:
    # Load and resize image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert image to base64 for HTML display
    img_base64 = get_image_base64(image)

    # Display image centered using HTML
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" alt="Uploaded MRI" width="150"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Predict button
    if st.button("🔍 Predict"):
        if not os.path.exists(MODEL_PATH):
            st.error("🚫 Model file not found. Please make sure 'model.h5' exists.")
        else:
            model = load_model(MODEL_PATH)
            resized = image.resize((240, 240))
            img_array = img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = prediction[predicted_index] * 100

            # Show result
            if predicted_class == "Tumor":
                st.error("⚠️ Tumor Detected")
            else:
                st.success("✅ No Tumor Detected")

            # Confidence bar
            st.markdown(f"**Confidence: {confidence:.2f}%**")
            st.progress(int(confidence))
