from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from decouple import config
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = config("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = config("ANTHROPIC_API_KEY")
os.environ["USER_AGENT"] = "The moose is loose" # Doesn't work!



template = """Question: {question}

Answer: (in one sentence please...)"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="codellama")

chain = prompt | model

result = chain.stream({"question": "Who is Trump"})

# print(result)
for chunk in result:
    print(chunk)