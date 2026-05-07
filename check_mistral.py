import os
import requests
from dotenv import load_dotenv

load_dotenv()

headers = {"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"}
r = requests.get("https://api.mistral.ai/v1/models", headers=headers, timeout=30)

print("STATUS:", r.status_code)
print("BODY:", r.text)