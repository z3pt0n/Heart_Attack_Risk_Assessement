# app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# Constants
INPUT_SIZE = 224
RISK_LEVELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Load models once (cache to avoid reloading every time)
@st.cache_resource
def load_models():
    eff_model = tf.keras.models.load_model('The_model.keras')
    resnet_model = tf.keras.models.load_model('The_model_resnet.keras')
    return eff_model, resnet_model

# Preprocessing function
def preprocess_image(image_file, model_type):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    image = np.array(image)

    if model_type == 'efficientnet':
        image = image / 255.0
    elif model_type == 'resnet':
        image = resnet_preprocess(image)

    return np.expand_dims(image, axis=0)

# UI
st.title("🧠 Diabetic Retinopathy Detection App")

# Upload section
uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose a model for prediction", ["EfficientNet", "ResNet50"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load models
    eff_model, resnet_model = load_models()
    selected_model = eff_model if model_choice == "EfficientNet" else resnet_model
    model_type = 'efficientnet' if model_choice == "EfficientNet" else 'resnet'

    # Preprocess and predict
    image = preprocess_image(uploaded_file, model_type)
    prediction = selected_model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    risk_level = RISK_LEVELS[predicted_class]

    # Show prediction
    st.subheader("🩺 Prediction Result")
    st.write(f"**Predicted Risk Level:** {risk_level}")
    st.write(f"**Model Raw Output:** {prediction.tolist()}")

    with st.expander("🔍 View Model Summary"):
        st.text(selected_model.summary(print_fn=lambda x: st.text(x)))
