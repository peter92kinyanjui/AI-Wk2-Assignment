import numpy as np
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the MNIST model
model = load_model('mnist_model.h5')  # Ensure you have your trained model saved as 'mnist_model.h5'

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Streamlit app title
st.title("MNIST Handwritten Digit Classifier")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of a handwritten digit (28x28 PNG/JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = plt.imread(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    image = image.reshape(1, 28, 28, 1) / 255.0  # Reshape and normalize

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Show the prediction
    st.write(f"Predicted Digit: {predicted_class}")
    
    
    model.save('mnist_model.h5')
