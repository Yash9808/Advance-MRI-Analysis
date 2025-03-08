import streamlit as st
import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from PIL import Image
import medspacy

# Load MedSpaCy model for medical entity extraction
nlp = medspacy.load()

# Load Pretrained MRI Analysis Model (Placeholder)
def load_mri_model():
    try:
        model = load_model("mri_model.h5")  # Replace with actual model file
        return model
    except Exception as e:
        st.error("Failed to load MRI model. Please ensure the model file is available.")
        st.stop()

# Process MRI Image and Detect Abnormalities
def analyze_mri(image, model):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    img_resized = img_resized.reshape(1, 128, 128, 1)
    prediction = model.predict(img_resized)
    predicted_class = np.argmax(prediction, axis=-1)  # For classification problems
    return predicted_class

# Extract key findings from report using MedSpaCy
def extract_findings(report_text):
    doc = nlp(report_text)
    # Extract medical entities such as disease, diagnosis, anatomical structure
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

# Streamlit UI
st.title("MRI Scan Analysis & Report Validation")

uploaded_image = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"])
uploaded_report = st.text_area("Paste Radiology Report Text")

if uploaded_image and uploaded_report:
    st.image(uploaded_image, caption="Uploaded MRI Scan", use_column_width=True)
    image = Image.open(uploaded_image)
    model = load_mri_model()
    
    # Analyze MRI
    prediction = analyze_mri(image, model)
    st.write(f"Predicted Condition: {prediction}")
    
    # Extract Findings from Report
    findings = extract_findings(uploaded_report)
    st.write("Extracted Findings from Report:", findings)
    
    # Highlight abnormalities
    highlighted_img = highlight_abnormalities(image)
    st.image(highlighted_img, caption="Highlighted Abnormalities", use_column_width=True)
    
    # Comparison
    st.write("Comparison of MRI vs. Report Findings")
    discrepancies = set([str(prediction)]) - set(findings)  # Convert prediction to string for comparison
    if discrepancies:
        st.write("Discrepancies Found:", discrepancies)
    else:
        st.write("MRI and Report Findings Match!")
else:
    st.warning("Please upload both an MRI scan and a radiology report.")
