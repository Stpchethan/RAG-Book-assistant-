from langchain_core.prompts import ChatPromptTemplate


def generate_query_variants(question: str, llm, num_queries: int = 3):
    """
    Generate multiple search queries from one user question.
    Used for Multi-Query Retrieval.
    """

    prompt = ChatPromptTemplate.from_template("""
You are an AI search query generator.

Create {num_queries} different search queries for the question below.
Return only the queries, one per line.

Question:
{question}
""")

    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "num_queries": num_queries
    })

    text = response.content if hasattr(response, "content") else str(response)

    queries = [
        q.strip("- ").strip()
        for q in text.split("\n")
        if q.strip()
    ]

    return queries[:num_queries]