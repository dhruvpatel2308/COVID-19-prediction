import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('my_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title('COVID-19 Detection from Chest X-rays')
st.write('Upload a chest X-ray image to get a prediction.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(224, 224))
    
    # Make a prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = ['COVID-19', 'Non-COVID', 'Normal']  # Update this based on your classes
    
    # Display the prediction
    st.write(f'Prediction: {class_labels[predicted_class]}')
    st.write(f'Confidence: {prediction[0][predicted_class]:.2f}')
