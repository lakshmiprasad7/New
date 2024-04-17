import os
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import PyPDF2

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

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("Text Extraction and MetaData & Summarization")

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
                    extracted_text = extract_text_from_image(image)
                    if extracted_text is None or len(extracted_text) == 0:
                        st.write("Unable to Detect Anything, use the AI Model for better results")

                    else:
                        st.write(extracted_text)

if __name__ == "__main__":
    main()
