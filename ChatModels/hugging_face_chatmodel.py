from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
import os

# ✅ Set your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# ✅ Use HuggingFaceEndpoint to configure the LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",  # For instruction-tuned models
    max_new_tokens=200,
    temperature=0.5,
)

# ✅ Now wrap it in ChatHuggingFace
chat = ChatHuggingFace(llm=llm)

# ✅ Run prompt
response = chat.invoke("What is the capital of India?")

print(response.content)
