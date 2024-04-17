import os
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import torch
import PyPDF2
from ultralytics import YOLO
import supervision as sv
import cv2


# Attempt to import the T5Tokenizer from transformers
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
except ImportError as e:
    st.error(f"RAM exhausted please relode the page to use the Webapp.")
    st.stop()



# Filter out specific warning categories
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Model and tokenizer loading
checkpoint = "models/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# Function to extract text from PDF using PyTesseract
def extract_text_from_pdf(pdf_path):
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    images = convert_from_path(pdf_path, output_folder=temp_dir)
    all_text = ''
    for img in images:
        text = pytesseract.image_to_string(img)
        all_text += text
    for image_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, image_file))
    os.rmdir(temp_dir)
    return all_text

# Function to extract metadata-like summary from PDF
def generate_metadata_summary(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        title = pdf_reader.metadata.title
        author = pdf_reader.metadata.author
        num_pages = len(pdf_reader.pages)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        summarization_length = len(text) // 6
        pipe_sum = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            max_length=summarization_length,
            min_length=50
        )
        summary = pipe_sum(text)
        summary_text = summary[0]['summary_text']
        metadata_summary = {
            "Title": title,
            "Author": author,
            "Number of Pages": num_pages,
            "Summary": summary_text
        }
    return metadata_summary



# YOLO Model loading
model_path = "models/YOLOv8_text_detection_model/best.pt"
yolo_model = YOLO(model_path)
class_names = {0: "text"}

# Function to perform text detection with YOLO and extract cropped images
import cv2

def text_detection_and_extraction(image_path):
    frame = cv2.imread(image_path)
    result = yolo_model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    cropped_images = []
    for detection in detections:
        bbox = list(detection[0])
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped_image = frame[y_min:y_max, x_min:x_max]

        # Convert cropped image to grayscale
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Invert colors
        inverted_image = cv2.bitwise_not(cropped_image_gray)

        # Apply thresholding to create binary image
        _, binary_image = cv2.threshold(inverted_image, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological operations for smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        smoothed_image = cv2.dilate(binary_image, kernel, iterations=1)
        smoothed_image = cv2.erode(smoothed_image, kernel, iterations=1)

        cropped_images.append(smoothed_image)
    return cropped_images





# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


def main():
    st.title("Text Extraction and MetaData & Summarization Using AI")

    uploaded_file = st.file_uploader("Upload your PDF file or Image", type=['pdf', 'png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        if uploaded_file.type == 'application/pdf':
            # PDF file
            if st.button("Extract Text and Generate Metadata (PDF)"):
                with st.spinner('Processing...'):
                    filepath = "pdfs/" + uploaded_file.name
                    with open(filepath, "wb") as temp_file:
                        temp_file.write(uploaded_file.read())

                    st.header("Extracted Text:")
                    extracted_text = extract_text_from_pdf(filepath)
                    st.write(extracted_text)

                    st.header("Metadata:")
                    metadata_summary = generate_metadata_summary(filepath)
                    for key, value in metadata_summary.items():
                        st.write(f"{key}: {value}")

        else:
            # Image file
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            if st.button("Extract Text (Image)"):
                with st.spinner('Processing...'):
                    st.header("Extracted Text:")
                    # Save the uploaded image temporarily
                    image_path = "temp_image.jpg"
                    image.save(image_path)
                    
                    # Perform text detection and extraction with YOLO
                    cropped_images = text_detection_and_extraction(image_path)

                    if len(cropped_images) == 0:
                        st.write("No text was detected in the uploaded image.")

                    else:

                        for i, cropped_image in enumerate(cropped_images):
                            st.subheader(f"Cropped Image {i+1}")
                            st.image(cropped_image, caption=f"Cropped Image {i+1}", use_column_width=True)

                            # Extract text from each cropped image
                            extracted_text = extract_text_from_image(cropped_image)
                            st.write(f"Text from Cropped Image {i+1}:")
                            st.write(extracted_text)

                    # Remove the temporary image file
                    os.remove(image_path)

if __name__ == "__main__":
    main()

