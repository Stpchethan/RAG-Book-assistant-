from langchain_mistralai import ChatMistralAI

from vector_store.manager import get_vectorstore
from vector_store.config import TOP_K, FETCH_K, LAMBDA_MULT

from retrievers.multiquery import generate_query_variants
from retrievers.reranker import rerank_documents
from retrievers.source_filter import filter_docs_by_source


def retrieve_with_multiquery(
    user_query: str,
    selected_source: str = "All Documents"
):
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3
    )

    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K,
            "lambda_mult": LAMBDA_MULT
        }
    )

    query_variants = generate_query_variants(
        question=user_query,
        llm=llm,
        num_queries=3
    )

    all_docs = []
    seen = set()

    for query_variant in query_variants:
        docs = retriever.invoke(query_variant)

        docs = filter_docs_by_source(
            docs,
            selected_source
        )

        for doc in docs:
            unique_key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content[:120]
            )

            if unique_key not in seen:
                seen.add(unique_key)
                all_docs.append(doc)

    ranked_docs = rerank_documents(
        user_query,
        all_docs,
        top_n=TOP_K
    )

    return ranked_docs