import os
from document_loaders.ingest_pipeline import ingest_pdf


def get_pdf_files(folder_path: str):
    pdf_files = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(folder_path, file))

    return pdf_files


def ingest_all_pdfs(folder_path: str = "document_loaders"):
    pdf_files = get_pdf_files(folder_path)

    results = []

    for pdf in pdf_files:
        print(f"\nIndexing: {pdf}")
        result = ingest_pdf(pdf)
        results.append({
            "file": pdf,
            "pages": result["pages_loaded"],
            "chunks": result["chunks_created"]
        })

    return results