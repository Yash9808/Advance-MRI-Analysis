import streamlit as st
import cv2
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

# MRI Model URLs
MRI_MODELS = {
    "Brain MRI": "https://huggingface.co/wizaye/MRI_LLM/resolve/main/brain_model.h5",
    "Spine MRI": "https://huggingface.co/spine-model/spine_model.h5",
    "Knee MRI": "https://huggingface.co/knee-model/knee_model.h5",
    "Cardiac MRI": "https://huggingface.co/cardiac-model/cardiac_model.h5",
    "Abdomen MRI": "https://huggingface.co/abdomen-model/abdomen_model.h5"
}

# MRI Condition Labels
MRI_LABELS = {
    "Brain MRI": ["Stroke", "Tumor", "Hemorrhage", "Fluid Buildup", "Normal"],
    "Spine MRI": ["Herniated Disc", "Degenerative Disease", "Spinal Stenosis", "Normal"],
    "Knee MRI": ["ACL Tear", "Meniscus Injury", "Arthritis", "Normal"],
    "Cardiac MRI": ["Cardiomyopathy", "Myocardial Infarction", "Heart Failure", "Normal"],
    "Abdomen MRI": ["Liver Disease", "Kidney Stones", "Tumor", "Normal"]
}

# Function to download the selected MRI model
def download_model(mri_type):
    model_path = f"{mri_type.replace(' ', '_').lower()}_model.h5"
    if not os.path.exists(model_path):
        st.warning(f"Downloading {mri_type} model... This may take a few minutes.")
        response = requests.get(MRI_MODELS[mri_type], stream=True)
        with open(model_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        st.success(f"{mri_type} model downloaded successfully!")
    return model_path

# Function to load the selected MRI model
def load_mri_model(mri_type):
    model_path = download_model(mri_type)
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load {mri_type} model: {e}")
        st.stop()

# Function to preprocess MRI images based on type
def preprocess_mri(image, mri_type, model):
    img = np.array(image.convert('RGB'))
    
    # Convert to grayscale (except for certain MRIs)
    if mri_type != "Knee MRI" and mri_type != "Cardiac MRI":  # Example: Knee and Cardiac MRIs may work better in RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    expected_shape = model.input_shape[1:3]
    img_resized = cv2.resize(img, expected_shape) / 255.0

    # Adjust channel count
    if model.input_shape[-1] == 3:
        img_resized = np.stack([img_resized] * 3, axis=-1)
    else:
        img_resized = img_resized.reshape(1, *expected_shape, 1)

    return np.expand_dims(img_resized, axis=0)

# Function to analyze MRI image
def analyze_mri(image, mri_type, model):
    processed_img = preprocess_mri(image, mri_type, model)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=-1)
    
    return predicted_class, processed_img

# Function to highlight abnormalities
def highlight_abnormalities(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted = img.copy()
    cv2.drawContours(highlighted, contours, -1, (255, 0, 0), 2)
    return highlighted

# Function to analyze report text and compare with AI prediction using BERT
def analyze_report(report_text, predicted_condition):
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    inputs = tokenizer(report_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    report_embedding = outputs.logits.squeeze().numpy()
    
    # Predicted condition to compare
    predicted_condition_str = str(predicted_condition).lower()
    
    # Convert predicted condition to embedding
    condition_embedding = tokenizer(predicted_condition_str, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        condition_outputs = model(**condition_embedding)
    
    condition_embedding = condition_outputs.logits.squeeze().numpy()

    # Calculate similarity score using cosine similarity
    similarity_score = cosine_similarity([report_embedding], [condition_embedding])
    confidence_score = round(similarity_score[0][0] * 10, 1)

    return confidence_score

# Streamlit UI
st.title("🧠 MRI Scan Analysis & Interactive Diagnosis")

# Select MRI Type
mri_type = st.selectbox("Select MRI Type:", list(MRI_MODELS.keys()))

# Select Question for Analysis
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

# Process image
if uploaded_image and question:
    model = load_mri_model(mri_type)
    image = Image.open(uploaded_image)

    # Get prediction and processed grayscale image
    prediction, processed_img = analyze_mri(image, mri_type, model)
    highlighted_img = highlight_abnormalities(image)

    # Display images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(uploaded_image, caption="Original MRI Scan", use_column_width=True)

    with col2:
        st.image(processed_img.squeeze(), caption="Processed MRI", use_column_width=True, channels="GRAY")

    with col3:
        st.image(highlighted_img, caption="Highlighted Abnormalities", use_column_width=True)

    # Display analysis result
    st.subheader("🔍 MRI Analysis Result")

    condition_labels = MRI_LABELS[mri_type]
    predicted_label = condition_labels[prediction[0]] if prediction[0] < len(condition_labels) else "Unknown Condition"
    
    st.write(f"Predicted Condition: **{predicted_label}**")

    # Generate bar chart
    fig, ax = plt.subplots()
    categories = condition_labels
    values = np.random.randint(10, 90, size=len(categories))  # Random values for demo

    ax.bar(categories, values, color=["red", "blue", "green", "orange", "gray"])
    ax.set_ylabel("Likelihood (%)")
    ax.set_title("Predicted MRI Findings")

    st.pyplot(fig)

    # Report comparison section
    st.subheader("📄 Report Analysis & Verification")
    report_text = st.text_area("Paste the radiology report here for AI comparison:")

    if report_text:
        confidence_score = analyze_report(report_text, predicted_label)

        # Display confidence score
        st.write(f"🔎 AI vs. Report Match Score: **{confidence_score}/10**")

        # Show progress bar
        st.progress(int(confidence_score))

else:
    st.warning("Please upload an MRI scan to start the analysis.")
