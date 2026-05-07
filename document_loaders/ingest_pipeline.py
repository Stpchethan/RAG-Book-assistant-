from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from vector_store.manager import get_embedding_model
from vector_store.config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from document_loaders.metadata_utils import enrich_documents_with_metadata
from document_loaders.document_registry import register_document


def ingest_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    chunks = enrich_documents_with_metadata(chunks, file_path)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    register_document(
        file_path=file_path,
        pages=len(docs),
        chunks=len(chunks)
    )

    return {
        "pages_loaded": len(docs),
        "chunks_created": len(chunks),
        "vectorstore": vectorstore
    }