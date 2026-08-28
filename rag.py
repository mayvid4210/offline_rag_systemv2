from pathlib import Path
import hashlib

import fitz


UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


# ------------------------------------
# Save Uploaded File
# ------------------------------------

def save_uploaded_file(uploaded_file):

    file_path = UPLOAD_FOLDER / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ------------------------------------
# Generate Document ID
# ------------------------------------

def get_document_id(file_path):

    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ------------------------------------
# Extract PDF Pages
# ------------------------------------

def extract_pages(file_path):

    pdf = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text()

        if text.strip():

            pages.append(
                {
                    "page_number": page_number,
                    "text": text
                }
            )

    pdf.close()

    return pages