from document_loaders.document_registry import load_registry


def get_available_documents():
    registry = load_registry()

    documents = ["All Documents"]

    for item in registry:
        file_path = item.get("file_path", "")
        file_name = file_path.replace("\\", "/").split("/")[-1]

        if file_name not in documents:
            documents.append(file_name)

    return documents