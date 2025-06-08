import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set up the page configuration
st.set_page_config(page_title="Heart Attack Risk Prediction", layout="centered")

# Title of the web app
st.title("Heart Attack Risk Prediction Using Retinal Imaging")

# Input size for image processing
INPUT_SIZE = 224

# Preprocess function for a single image
def preprocess_single_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = preprocess_input(image)  
    return np.expand_dims(image, axis=0)

# Function to load the model and predict
def predict_image(model_path, image_path):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    image = preprocess_single_image(image_path)
    
    # Model summary (for debugging purposes, can be commented out if not needed)
    st.write("Model Summary:")
    model.summary()

    # Predict the class of the image
    prediction = model.predict(image)
    st.write("Prediction Raw Output:", prediction)

    # Determine the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class to risk level
    risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    risk_level = risk_levels[predicted_class]

    # Return the risk level
    return risk_level

# File uploader widget for users to upload an image
uploaded_image = st.file_uploader("Upload an Image of Retina", type=["jpg", "jpeg", "png"])

# Dropdown for model selection
model_choice = st.selectbox("Select the Model", ["EfficientNet", "ResNet"])

# Display the uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict using the selected model
    if model_choice == "EfficientNet":
        model_path = r"C:\Users\harsh\Desktop\Projects\Heart_Attack_Mgmt\The_model.keras"
    else:
        model_path = r"C:\Users\harsh\Desktop\Projects\Heart_Attack_Mgmt\The_model_resnet.keras"
    
    if st.button("Predict"):
        # Run prediction on the uploaded image
        risk_level = predict_image(model_path, image)

        # Show the result
        st.write("Predicted Risk Level of Heart Attack:", risk_level)











# app.py

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# # Constants
# INPUT_SIZE = 224
# RISK_LEVELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# # Load models once (cache to avoid reloading every time)
# @st.cache_resource
# def load_models():
#     eff_model = tf.keras.models.load_model('The_model.keras')
#     resnet_model = tf.keras.models.load_model('The_model_resnet.keras')
#     return eff_model, resnet_model

# # Preprocessing function
# def preprocess_image(image_file, model_type):
#     image = Image.open(image_file).convert('RGB')
#     image = image.resize((INPUT_SIZE, INPUT_SIZE))
#     image = np.array(image)

#     if model_type == 'efficientnet':
#         image = image / 255.0
    # elif model_type == 'resnet':
    #     image = resnet_preprocess(image)

    # return np.expand_dims(image, axis=0)

# # UI
# st.title("🧠 Diabetic Retinopathy Detection App")

# # Upload section
# uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])
# model_choice = st.selectbox("Choose a model for prediction", ["EfficientNet", "ResNet50"])

# if uploaded_file is not None:
#     # Display image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     # Load models
#     eff_model, resnet_model = load_models()
#     selected_model = eff_model if model_choice == "EfficientNet" else resnet_model
#     model_type = 'efficientnet' if model_choice == "EfficientNet" else 'resnet'

#     # Preprocess and predict
#     image = preprocess_image(uploaded_file, model_type)
#     prediction = selected_model.predict(image)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     risk_level = RISK_LEVELS[predicted_class]

#     # Show prediction
#     st.subheader("🩺 Prediction Result")
#     st.write(f"**Predicted Risk Level:** {risk_level}")
#     st.write(f"**Model Raw Output:** {prediction.tolist()}")

#     with st.expander("🔍 View Model Summary"):
#         st.text(selected_model.summary(print_fn=lambda x: st.text(x)))
