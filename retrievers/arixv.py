from langchain_community.retrievers import ArxivRetriever
from langchain_community.vectorstores import Chroma
#Create retriever 

retriever = ArxivRetriever(
load_max_docs= 2 ,
load_all_available_meta=True


)

#query Axis

docs = retriever.invoke("neural networks")

#print Results 

for i , doc in enumerate(docs):
    print(f"\nResult{i+1}")
    print("Title:", doc.metadata.get("Title:"))
    print("Authors:", doc.metadata.get("Authors:"))
    print("Summary:", doc.page_content)
    










