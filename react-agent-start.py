# Import the required modules
import wikipedia
import pprint
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# ========================= REACT AGENT TOOLS START ==========================================

# Define the tool for DuckDuckGo search
@tool
def search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo for a given query and return the first result snippet."""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)  # Customize max_results as needed
    # results = wrapper.results(query)
    results = wrapper.results(query=query, max_results=1)
    if results:
        result = results[0]  # Get the first result
        return f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}"
    return "No results found."

# Successfully tested the tool
# duck_result = search_duckduckgo.invoke("Donald Trump")
# print("DUCK RESULT TEST", duck_result)

# Import the required modules for Wikipedia search
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

# Define the Wikipedia search tool
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a given query and return a summary."""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

# Successfully tested the tool
# wiki_result = search_wikipedia.invoke("Donald Trump")
# print("WIKI RESULT TEST", wiki_result)

# ========================= REACT AGENT TOOLS ENDS ==========================================

from decouple import config
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = config("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

# ========================= REACT AGENT CORE STARTS ==========================================

# Import LangGraph prebuilt utilities
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama  # <-- Modern Ollama wrapper
from langchain_openai import ChatOpenAI
import pprint

# Set up the language model
model = ChatOllama(model="MFDoom/deepseek-coder-v2-tool-calling:16b").bind_tools([search_duckduckgo, search_wikipedia])
# model = ChatOllama(model="deepseek-coder-v2:16b").bind_tools([search_duckduckgo, search_wikipedia])
# model = ChatOpenAI(model="deepseek-coder-v2:16b", temperature=0.5)
# model = ChatOpenAI(model="gpt-4o", temperature=0.5)

# List of tools we created
tools = [search_duckduckgo, search_wikipedia]

# Create a LangGraph agent using the tools
new_react_agent = create_react_agent(model, tools)

# Define a simple user query to invoke the agent
query = "tell me about U.S. Election 2024"

# Invoke the agent with the query wrapped in a message
messages = new_react_agent.invoke({"messages": [("human", query)]})

output = messages["messages"][-1].content
print(output)

# ========================= REACT AGENT CORE ENDS ============================================

