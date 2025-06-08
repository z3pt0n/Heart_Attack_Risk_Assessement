#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Corrected input size for EfficientNetB0
INPUT_SIZE = 224

# Preprocess function for a single image
def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


# Load the trained model
model = tf.keras.models.load_model('The_model.keras')

# Image path for prediction
image_path = r"C:\Users\harsh\Desktop\Projects\Heart_Attack_Mgmt\uploads\1084_right.jpeg"

# Preprocess the image
image = preprocess_single_image(image_path)

# Summarize the model
print("Model Summary:")
model.summary()

# Predict the class of the image
prediction = model.predict(image)
print("Prediction Raw Output:", prediction)

# Determine the predicted class
predicted_class = np.argmax(prediction, axis=1)[0]

# Map predicted class to risk level
risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
risk_level = risk_levels[predicted_class]

# Print the predicted risk level
print("Predicted risk level of heart attack:", risk_level)


# In[7]:


import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

INPUT_SIZE = 224

def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = preprocess_input(image)  
    return np.expand_dims(image, axis=0)

model = tf.keras.models.load_model(r'C:\Users\harsh\Desktop\Projects\Heart_Attack_Mgmt\The_model_resnet.keras')  

image_path = r"C:\Users\harsh\Desktop\Projects\Heart_Attack_Mgmt\uploads\0f96c358a250.png"
image = preprocess_single_image(image_path)

print("Model Summary:")
model.summary()

prediction = model.predict(image)
print("Prediction Raw Output:", prediction)

predicted_class = np.argmax(prediction, axis=1)[0]  

risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
risk_level = risk_levels[predicted_class]

print("Predicted risk level of heart attack:", risk_level)

