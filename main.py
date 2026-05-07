from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from retrievers.citation_builder import format_citations_block
from vector_store.manager import get_llm
from retrievers.hybrid_retriever import retrieve_with_multiquery

from vector_store.chat_history import save_message

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant for document question answering.

Rules:
1. Use only the provided context.
2. If the answer is not present in the context, say:
   "I could not find the answer in the document."
3. Answer clearly and accurately.
4. At the end, include the citations section.
"""),
    ("human", """
Context:
{context}

Question:
{question}

Citations:
{citations}
""")
])


def main():
    query = "Your question"

    docs = retrieve_with_multiquery(query)

    print(docs)

if __name__ == "__main__":
    main()









def main():
    llm = get_llm()

    print("RAG system ready.")
    print("Type 0 to exit.")

    while True:
        query = input("\nYou: ").strip()

        if query == "0":
            print("Goodbye.")
            break

        docs = retrieve_with_multiquery(query)

        if not docs:
            print("AI: I could not find the answer in the document.")
            continue

        context = "\n\n".join(doc.page_content for doc in docs)
        citations = format_citations_block(docs)

        final_prompt = prompt.format(
            context=context,
            question=query,
            citations=citations
        )

        response = llm.invoke(final_prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        print(f"\nAI: {answer_text}\n")

        save_message("user", query)
        save_message("assistant", answer_text)


if __name__ == "__main__":
    main()