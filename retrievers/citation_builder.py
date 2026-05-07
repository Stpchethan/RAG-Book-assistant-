def build_citations(docs):
    citations = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "N/A")
        citations.append(f"{source} - page {page}")

    return list(dict.fromkeys(citations))


def format_citations_block(docs):
    citations = build_citations(docs)

    if not citations:
        return ""

    return "\n".join(f"- {c}" for c in citations)