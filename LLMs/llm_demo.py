from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
groq_api_key = os.getenv("GROQ_API_KEY")

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)


result = llm.invoke("What is the capital of India")

print(result)

