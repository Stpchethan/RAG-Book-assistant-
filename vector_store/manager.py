from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

from vector_store.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)

_embedding_model = None
_llm = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
    return _embedding_model


def get_vectorstore():
    embedding_model = get_embedding_model()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            mistral_api_key=MISTRAL_API_KEY
        )
    return _llm