def keyword_overlap_score(query: str, text: str) -> int:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    return len(query_words.intersection(text_words))


def rerank_documents(query: str, docs, top_n: int = 4):
    scored = []

    for doc in docs:
        score = keyword_overlap_score(query, doc.page_content)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]