import io # work with bytes in memory
import fitz # PyMuPDF for PDFs
import docx # python-docx

# TXT extractor
def extract_text_from_txt(uploaded_file):
    uploaded_file.seek(0)
    return uploaded_file.read().decode("utf-8")

# PDF extractor
def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    for page_num, page in enumerate(pdf_document, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page": page_num,
                "text": text
            })

    return pages

# DOCX extractor
def extract_text_from_docx(uploaded_file):
    uploaded_file.seek(0)
    doc = docx.Document(io.BytesIO(uploaded_file.read()))
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text

# determine file extension and extract content
def extract_text_from_file(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()

    if suffix == "txt":
        return {
            "type": "txt",
            "content": extract_text_from_txt(uploaded_file)
        }

    if suffix == "pdf":
        return {
            "type": "pdf",
            "content": extract_text_from_pdf(uploaded_file)
        }

    if suffix == "docx":
        return {
            "type": "docx",
            "content": extract_text_from_docx(uploaded_file)
        }

    return None