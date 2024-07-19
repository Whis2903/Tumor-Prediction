import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Loading the Trained Model
model = tf.keras.models.load_model('BrainTumor_CNN.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resizing the image to the input size requirement
    image = image.resize((64, 64))  
  
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions


st.title("Brain Tumor Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if st.button('Predict'):
        st.write("Classifying...")
        predictions = predict(image)
    
        class_idx = np.argmax(predictions)

        class_names = ["No tumor detected", "Tumor detected"]
        result = class_names[class_idx]

        st.write(result)
