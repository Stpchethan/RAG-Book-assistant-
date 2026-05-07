import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

print("KEY FOUND:", bool(os.getenv("MISTRAL_API_KEY")))

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    timeout=30,
)

response = llm.invoke("Say hello in one line.")
print(response.content)