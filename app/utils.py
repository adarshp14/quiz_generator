import fitz  # PyMuPDF for PDF processing
from docx import Document

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()
