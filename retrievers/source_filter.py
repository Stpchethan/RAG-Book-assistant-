def filter_docs_by_source(docs, selected_source: str):
    if selected_source == "All Documents":
        return docs

    filtered_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        if source == selected_source:
            filtered_docs.append(doc)

    return filtered_docs