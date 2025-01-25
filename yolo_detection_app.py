import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import cv2
import fitz  # PyMuPDF
import numpy as np
from model_loader import load_model


import pytesseract

# Set Tesseract command for Windows
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Step 1: Load the model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check your internet connection and try again.")
    st.stop()

# Streamlit App
st.markdown("""
    <div style="text-align: center;">
        <h2 style='color: #dc7ef3; font-size: 15px;'>Rajas Daryapurkar</h2>
        <h1 style='color: #8cf3ea; font-size: 60px;'>Document Segmenting and Text Retrival</h1>
    </div>
""", unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])

def add_padding(image, bbox, padding=10):
    """Adds padding to the bounding box."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    return x1, y1, x2, y2

# Function to preprocess image before passing to Tesseract
def preprocess_image(image):
    """Preprocess the image to enhance OCR accuracy."""
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to improve contrast
    _, thresholded_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
    
    # Optionally apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)
    
    return blurred_image

# Function to process the image and perform OCR
def process_image(image, model, page_num=None):
    """Processes an image, detects text, and performs OCR."""
    # Ensure image is a numpy array
    image_array = np.array(image)
    
    # Preprocess image for better OCR accuracy
    preprocessed_image = preprocess_image(image_array)
    
    # Ensure preprocessed image is 3-channel for YOLO
    if len(preprocessed_image.shape) == 2:
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    
    # Detect text regions using the model
    results = model(preprocessed_image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Objects", use_container_width=True)

    # Collect all detected regions with their coordinates and text
    detected_regions = []
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1, x2, y2 = add_padding(np.array(image), (x1, y1, x2, y2))

        # Crop the region from the image
        cropped_region = np.array(image)[y1:y2, x1:x2]

        # Ensure that the cropped region has valid dimensions
        if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
            # Perform OCR with Tesseract
            custom_config = r'--psm 6'  # Treats the image as a single block of text
            ocr_text = pytesseract.image_to_string(cropped_region, config=custom_config)
            detected_regions.append((y1, x1, ocr_text.strip()))  # Include y1, x1 for sorting

    # Sort detected regions by position (top-to-bottom, left-to-right)
    detected_regions.sort(key=lambda region: (region[0], region[1]))

    # Combine all extracted text into a single block in the correct order
    combined_text = "\n\n".join([text for _, _, text in detected_regions])

    # Display the combined text
    st.text_area(
        "Extracted Text",
        combined_text,
        height=300,  # Adjust the height as needed
    )



# In your main block, pass the page number when processing PDFs
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type in ["jpg", "jpeg", "png"]:
        # Process uploaded image files
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        process_image(image, model, page_num=0)

    elif file_type == "pdf":
        # Handle PDF files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_path = temp_pdf_file.name
            temp_pdf_file.write(uploaded_file.getvalue())

        try:
            # Open the PDF document
            doc = fitz.open(temp_pdf_path)

            for i in range(doc.page_count):
                # Process each page of the PDF
                page = doc.load_page(i)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(img, caption=f"Page {i+1}", use_container_width=True)
                process_image(img, model=model, page_num=i + 1)  # Pass page number to the function
        finally:
            # Ensure the temporary file is deleted
            os.remove(temp_pdf_path)
