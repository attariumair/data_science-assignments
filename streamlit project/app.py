
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load Model
model = load_model("digit_model.h5")

# App Title
st.title("🧠 Handwritten Digit Recognition App")
st.write("Upload an image of a handwritten digit (0–9), and the model will predict it.")

# Image Upload
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # Convert to grayscale
    img_gray = ImageOps.grayscale(image)

    # Resize to 28x28
    img_resized = img_gray.resize((28, 28))

    # Convert to array
    img_array = np.array(img_resized)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader("🔍 Prediction")
    st.success(f"Predicted Digit: **{predicted_digit}**")
