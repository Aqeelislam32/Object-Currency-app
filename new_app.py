import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from PIL import Image

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Function to predict the class and confidence
def predict_currency(image):
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Streamlit App
st.title("Currency Detection App")

# Sidebar options
option = st.sidebar.selectbox("Choose an input method:", ["Webcam", "IP CCTV", "Upload Image"])

# Set up the webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set higher resolution for better clarity
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
start_webcam = st.button("Start Webcam", key="start_webcam")
stop_webcam = st.button("Stop Webcam", key="stop_webcam")

if start_webcam:
    st.text("Webcam is running...")
    frame_window = st.image([])  # Placeholder for the video feed
    
    while True:
        ret, image = camera.read()
        if not ret:
            st.warning("Failed to grab frame from webcam.")
            break

        # Resize image for the model
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Display the live video feed
        frame_window.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Prepare image for prediction
        input_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
        input_image = (input_image / 127.5) - 1  # Normalize the image

        # Model prediction
        prediction = model.predict(input_image)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        st.write(f"Detected: {class_name} | Confidence: {confidence_score * 100:.2f}%")

        # Stop the loop if the Stop Webcam button is pressed
        if stop_webcam:
            break

        camera.release()
        st.text("Webcam stopped.")

elif option == "IP CCTV":
    st.subheader("Live IP CCTV Prediction")
    ip_url = st.text_input("Enter IP camera URL:", "http://your_ip_camera_url/stream")
    start_button = st.button("Start IP CCTV")

    if start_button and ip_url:
        camera = cv2.VideoCapture(ip_url)
        st_frame = st.empty()

        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access the IP camera.")
                break

            # Show the CCTV feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")

            # Preprocess and predict
            processed_frame = preprocess_image(frame)
            class_name, confidence = predict_currency(processed_frame)

            st.write(f"Prediction: {class_name}, Confidence: {confidence * 100:.2f}%")

            # Break if user stops the app
            if st.button("Stop IP CCTV"):
                break

        camera.release()

elif option == "Upload Image":
    st.subheader("Image Upload Prediction")
    uploaded_file = st.file_uploader("Upload an image of currency:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL Image to OpenCV format
        image = np.array(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        class_name, confidence = predict_currency(processed_image)

        st.write(f"Prediction: {class_name}, Confidence: {confidence * 100:.2f}%")
 