import hashlib
from pathlib import Path

import fitz

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
MAX_PDF_BYTES = 25 * 1024 * 1024


def get_document_id(data):
    return hashlib.sha256(data).hexdigest()


def save_uploaded_file(data, document_id):
    if get_document_id(data) != document_id:
        raise ValueError("PDF content does not match its document ID.")
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    path = UPLOAD_FOLDER / f"{document_id}.pdf"
    if path.exists():
        if get_document_id(path.read_bytes()) != document_id:
            raise ValueError("Stored PDF is damaged; preserve it and investigate before retrying.")
        return path
    file = path.open("xb")
    try:
        with file:
            file.write(data)
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


def extract_pages(data):
    if not data or len(data) > MAX_PDF_BYTES:
        raise ValueError("Select a nonempty PDF no larger than 25 MB.")
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported. Upload an unlocked copy.")
            if len(pdf) > 1000:
                raise ValueError("PDFs must contain at most 1,000 pages.")
            pages = []
            characters = 0
            for number, page in enumerate(pdf, start=1):
                text = page.get_text()
                characters += len(text)
                if characters > 2_000_000:
                    raise ValueError("PDF contains too much text (limit: 2 million characters).")
                if text.strip():
                    pages.append({"page_number": number, "text": text})
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError("Cannot read this PDF. Check that it is valid and not damaged.") from error
    if not pages:
        raise ValueError("No extractable text found. Scanned PDFs require OCR, which is not supported.")
    return pages


def delete_uploaded_file(document_id, document_name):
    # Legacy originals are removed only after their bytes prove their identity.
    managed = UPLOAD_FOLDER / f"{document_id}.pdf"
    if managed.exists():
        if get_document_id(managed.read_bytes()) != document_id:
            raise ValueError("Stored original has a different hash; it was preserved for inspection.")
        managed.unlink()
    legacy = UPLOAD_FOLDER / Path(document_name).name
    if legacy.is_file() and get_document_id(legacy.read_bytes()) == document_id:
        legacy.unlink()
