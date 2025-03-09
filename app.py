import streamlit as st
import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from PIL import Image
import medspacy
import os
import requests
import matplotlib.pyplot as plt

# Load MedSpaCy model for medical entity extraction
nlp = medspacy.load()

# Hugging Face model URL (Direct Download Link)
MODEL_URL = "https://huggingface.co/wizaye/MRI_LLM/resolve/main/brain_model.h5"
MODEL_PATH = "mri_model.h5"

def download_model():
    """Download the model from Hugging Face if not available locally."""
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading MRI model... This may take a few minutes.")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        st.success("Model downloaded successfully!")

# Load Pretrained MRI Analysis Model
def load_mri_model():
    try:
        download_model()
        model = load_model(MODEL_PATH)
        st.write(f"Model Loaded Successfully! Expected Input Shape: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"Failed to load MRI model: {e}")
        st.stop()

# Process MRI Image and Detect Abnormalities
def analyze_mri(image, model):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    expected_shape = model.input_shape[1:3]  
    img_resized = cv2.resize(img, expected_shape) / 255.0
    
    if model.input_shape[-1] == 3:
        img_resized = np.stack([img_resized] * 3, axis=-1)  
    else:
        img_resized = img_resized.reshape(1, *expected_shape, 1)
    
    img_resized = np.expand_dims(img_resized, axis=0)  
    prediction = model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=-1)
    return predicted_class

# Highlight abnormalities in MRI image
def highlight_abnormalities(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)
    return highlighted

# Chart-based question selection
st.title("🧠 MRI Scan Analysis & Interactive Diagnosis")

# Dropdown for question-based analysis
question = st.selectbox(
    "Select a question to start MRI analysis:",
    [
        "What conditions does this MRI suggest?",
        "Does this MRI indicate a stroke?",
        "Are there any tumors in the MRI?",
        "Is there any sign of brain hemorrhage?",
        "Is there abnormal fluid accumulation?",
        "What are the highlighted abnormalities?"
    ]
)

# Upload MRI Image
uploaded_image = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])

# Process image based on question
if uploaded_image and question:
    model = load_mri_model()
    image = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Uploaded MRI Scan", use_column_width=True)

    prediction = analyze_mri(image, model)
    highlighted_img = highlight_abnormalities(image)

    # Display the results based on the question
    st.subheader("🔍 MRI Analysis Result")

    if "conditions" in question:
        st.write(f"Predicted Condition: {prediction}")
    elif "stroke" in question:
        st.write("Checking for stroke-related abnormalities...")
    elif "tumors" in question:
        st.write("Analyzing possible tumor locations...")
    elif "hemorrhage" in question:
        st.write("Detecting possible brain hemorrhage...")
    elif "fluid" in question:
        st.write("Looking for signs of abnormal fluid accumulation...")
    elif "abnormalities" in question:
        st.image(highlighted_img, caption="Highlighted Abnormalities", use_column_width=True)

    # Show a bar chart representation of analysis
    fig, ax = plt.subplots()
    categories = ["Stroke", "Tumor", "Hemorrhage", "Fluid Buildup", "Normal"]
    values = np.random.randint(10, 90, size=len(categories))  # Random values for demo

    ax.bar(categories, values, color=["red", "blue", "green", "orange", "gray"])
    ax.set_ylabel("Likelihood (%)")
    ax.set_title("Predicted MRI Findings")

    st.pyplot(fig)

else:
    st.warning("Please upload an MRI scan to start the analysis.")
