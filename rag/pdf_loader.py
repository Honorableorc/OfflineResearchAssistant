from pypdf import PdfReader


def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    documents = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        documents.append(
            {
                "page": page_num + 1,
                "text": text if text else ""
            }
        )

    return documents

