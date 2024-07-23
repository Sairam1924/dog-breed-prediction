import streamlit as st
import pickle
from PIL import Image
import numpy as np
import io
import cv2 


# #aading innomatics logo
# st.image(r"E:\innomatics\logo.png", width=200)

# Load the logistic regression model
model_path = r"model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess the image and make prediction
def predict(image):
    # Preprocess the image as required by your model
    
    
    image = np.resize(image,(150,150))
    # image = np.resize(image,(30, 30))  # Example: Resize to (30, 30) if your model needs that
    image_array = np.array(image).flatten()  # Example: Flatten the image 
    
    
    # Make prediction
    prediction = model.predict([image_array])
    return prediction

# Streamlit UI
st.title('Image Classification App')
st.write('Upload an image and the model will predict the outcome.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    prediction = predict(image)
    
    # Display the prediction
    st.write(f'Prediction: {prediction[0]}')
