import os


def enrich_documents_with_metadata(chunks, file_path: str):
    source_name = os.path.basename(file_path)

    for idx, doc in enumerate(chunks):
        doc.metadata["source"] = source_name
        doc.metadata["chunk_id"] = idx + 1

        if "page" not in doc.metadata:
            doc.metadata["page"] = "N/A"

    return chunks