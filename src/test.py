from langchain_google_vertexai import ChatVertexAI

from dotenv import load_dotenv
import os

# Load environment variables from the specified .env file
load_dotenv(dotenv_path="D:\Zikra LLC\customer-success-draft\.env.example")
print(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
print(os.getenv("GOOGLE_CLOUD_LOCATION"))

print(os.getenv("GOOGLE_CLOUD_PROJECT"))

chat = ChatVertexAI(model="gemini-2.0-flash")
resp = chat.invoke("What's the weather like on Mars?")
print(resp.content)
