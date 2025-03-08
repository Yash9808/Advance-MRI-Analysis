import streamlit as st
import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from PIL import Image
import medspacy
import os
import requests

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
        download_model()  # Ensure the model is downloaded
        model = load_model(MODEL_PATH)
        
        # Print model input shape for debugging
        st.write(f"Model Loaded Successfully! Expected Input Shape: {model.input_shape}")
        
        return model
    except Exception as e:
        st.error(f"Failed to load MRI model: {e}")
        st.stop()

# Process MRI Image and Detect Abnormalities
def analyze_mri(image, model):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize image to model's expected input shape
    expected_shape = model.input_shape[1:3]  # Get height & width from model
    img_resized = cv2.resize(img, expected_shape) / 255.0
    
    # Adjust channels based on model expectations (ensure it has 3 channels)
    if model.input_shape[-1] == 3:  # If model expects RGB
        img_resized = np.stack([img_resized] * 3, axis=-1)  # Convert to 3-channel image
    else:
        img_resized = img_resized.reshape(1, *expected_shape, 1)  # Keep 1 channel
    
    # Add batch dimension (for single image, batch size = 1)
    img_resized = np.expand_dims(img_resized, axis=0)  # shape becomes (1, 299, 299, 3) for the model
    
    # Prediction
    prediction = model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=-1)
    return predicted_class

# Extract key findings from report using MedSpaCy
def extract_findings(report_text):
    doc = nlp(report_text)
    findings = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "TREATMENT", "ANATOMICAL_STRUCTURE"]]
    return findings

# Highlight abnormalities in MRI image
def highlight_abnormalities(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)
    return highlighted

# Function to handle the chat and provide more informative responses
def handle_chat(user_input):
    # Detect keywords related to medical conditions
    if "stroke" in user_input.lower():
        return (
            "Stroke occurs when blood flow to part of the brain is interrupted, causing brain cell damage. "
            "There are two main types of strokes:\n"
            "- Ischemic Stroke: A blockage in a blood vessel reduces blood flow to the brain.\n"
            "- Hemorrhagic Stroke: Caused by bleeding into the brain.\n"
            "MRI can identify damaged areas in the brain caused by both types of stroke."
        )
    elif "brain hemorrhage" in user_input.lower():
        return (
            "A brain hemorrhage refers to bleeding within the brain. Types include:\n"
            "- Subdural Hematoma: Bleeding between the brain and its outer covering.\n"
            "- Intracerebral Hemorrhage: Bleeding within the brain tissue.\n"
            "MRI scans can help identify and assess the extent of bleeding in the brain."
        )
    elif "condition" in user_input.lower() or "disease" in user_input.lower():
        return (
            "The MRI scan can detect various conditions, including:\n"
            "- Brain Tumors (benign or malignant)\n"
            "- Stroke and Brain Hemorrhages\n"
            "- Alzheimer's Disease\n"
            "- Epilepsy\n"
            "- Multiple Sclerosis (MS)\n"
            "- Infections like encephalitis or abscesses\n"
            "- Parkinson's Disease\n"
            "- Hydrocephalus (fluid buildup in the brain)\n"
            "- Traumatic Brain Injury (TBI)\n"
            "Would you like more details on any of these conditions?"
        )
    elif "abnormality" in user_input.lower():
        return "The highlighted abnormalities are based on algorithmic detection in the MRI. Would you like more details on these?"
    elif "diagnosis" in user_input.lower():
        return "Based on the model's analysis, the MRI could show brain-related conditions, but further clinical evaluation is needed for a definitive diagnosis."
    else:
        return "I am here to assist with MRI analysis. Please ask any questions related to the MRI scan or the report."

# Adding Background GIF using CSS
st.markdown("""
    <style>
        .reportview-container {
            background-image: url('https://raw.githubusercontent.com/Yash9808/MRI-Image-Report-Analysis/main/MRI_brain_scan.gif');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            position: absolute;
            width: 100%;
            z-index: -1;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("MRI Scan Analysis & Report Validation")

# Allow multiple images to be uploaded
uploaded_images = st.file_uploader("Upload MRI Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
uploaded_report = st.text_area("Paste Radiology Report Text")

# Chat functionality
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chatbox UI
st.header("Chat with the MRI Analysis Assistant")
user_message = st.text_input("Ask me anything about the MRI Scan or Report:")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    bot_response = handle_chat(user_message)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display chat messages
for message in st.session_state.messages:
    if message['role'] == "user":
        st.markdown(f"**User**: {message['content']}")
    else:
        st.markdown(f"**Assistant**: {message['content']}")

if uploaded_images and uploaded_report:
    # Loop through all uploaded images
    model = load_mri_model()
    
    for uploaded_image in uploaded_images:
        st.image(uploaded_image, caption=f"Uploaded MRI Scan", use_column_width=True)
        image = Image.open(uploaded_image)
        
        # Analyze each MRI image
        prediction = analyze_mri(image, model)
        st.write(f"Predicted Condition for this MRI: {prediction}")
        
        # Highlight abnormalities
        highlighted_img = highlight_abnormalities(image)
        st.image(highlighted_img, caption="Highlighted Abnormalities", use_column_width=True)
    
    # Extract Findings from Report
    findings = extract_findings(uploaded_report)
    st.write("Extracted Findings from Report:", findings)
    
    # Comparison
    st.write("Comparison of MRI vs. Report Findings")
    discrepancies = set([str(prediction)]) - set(findings)  # Convert prediction to string for comparison
    if discrepancies:
        st.write("Discrepancies Found:", discrepancies)
    else:
        st.write("MRI and Report Findings Match!")
else:
    st.warning("Please upload both MRI scans and a radiology report.")
