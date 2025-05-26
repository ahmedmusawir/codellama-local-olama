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
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)  # Customize max_results as needed
    # results = wrapper.results(query)
    results = wrapper.results(query=query, max_results=5)
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
# TOOLS - 2 tries but wrong answer both times - USED: DUCK DUCK GO - TIME: 39sec + 15GB Ram
# model = ChatOllama(model="call518/deepseek-r1-tool-calling:7b").bind_tools([search_duckduckgo])

# TOOLS - 2 tries - one correct and one wrong answers - USED: DUCK DUCK GO - TIME: 19sec + 23GB Ram
# model = ChatOllama(model="MFDoom/deepseek-r1-tool-calling:14b").bind_tools([search_duckduckgo])

# TOOLS - One line Accurate Answer - took too long - USED: DUCK DUCK GO - TIME: 42.20sec + 32GB Ram
# model = ChatOllama(model="MFDoom/deepseek-r1-tool-calling:32b").bind_tools([search_duckduckgo])
# model = ChatOllama(model="MFDoom/deepseek-r1-tool-calling:32b").bind_tools([search_duckduckgo, search_wikipedia])

# TOOLS - wrong anser ... joe biden - USED: DUCK DUCK GO - TIME: 32sec + 25GB Ram
# model = ChatOllama(model="hhao/qwen2.5-coder-tools:14b").bind_tools([search_duckduckgo])
#model = ChatOllama(model="hhao/qwen2.5-coder-tools:14b").bind_tools([search_duckduckgo, search_wikipedia])

# TOOLS - to the point answer - USED: DUCK DUCK GO - TIME: 6sec + 18GB Ram
# model = ChatOllama(model="MFDoom/deepseek-coder-v2-tool-calling:16b").bind_tools([search_duckduckgo])
# model = ChatOllama(model="MFDoom/deepseek-coder-v2-tool-calling:16b").bind_tools([search_duckduckgo, search_wikipedia])

# NEVER USE - SIMPLY FAILED - HAD TO BE STOPPED AFTER 5MIN - USED: NEVER PRODUCED RESULTS - TIME: 5MIN + 35GB Ram
# model = ChatOllama(model="ishumilin/deepseek-r1-coder-tools-tuned:14b").bind_tools([search_duckduckgo])

#FAILED ON TOOLS - no answers for a long time taking 33gb ram - multiple tries and failed each time to use tools
# model = ChatOllama(model="tom_himanen/deepseek-r1-roo-cline-tools:14b").bind_tools([search_duckduckgo, search_wikipedia])

#FAILED ON TOOLS - no answers for a long time taking 33gb ram - had to manually kill it
# model = ChatOllama(model="ishumilin/deepseek-r1-coder-tools:14b").bind_tools([search_duckduckgo, search_wikipedia])

# FAILED ON TOOLS - worst model so far - USED: DUCK DUCK GO - TIME: 12sec + 23GB Ram
# model = ChatOllama(model="llama3-groq-tool-use:8b").bind_tools([search_duckduckgo])


# NO TOOLS
# model = ChatOllama(model="deepseek-coder-v2:16b").bind_tools([search_duckduckgo, search_wikipedia])
# EXAMPLES
# model = ChatOpenAI(model="deepseek-coder-v2:16b", temperature=0.5)
# model = ChatOpenAI(model="gpt-4o", temperature=0.5)

# List of tools we created
tools = [search_duckduckgo]
# tools = [search_duckduckgo, search_wikipedia]

# Create a LangGraph agent using the tools
new_react_agent = create_react_agent(model, tools, prompt="Your are a helpful agent that can use tools to answer. Specially, you wanna be current when you give informative answers")

# Define a simple user query to invoke the agent
# query = "tell me about the result of U.S. Election 2024"
# query = "tell me about the result of U.S. Election 2024, who won?"
query = "Who won the U.S. Election in 2024?"

# Invoke the agent with the query wrapped in a message
messages = new_react_agent.invoke({"messages": [("human", query)]})

output1 = messages["messages"]
print("DETAILS OF ANSWER", output1)
print("===========================================================")
output2 = messages["messages"][-1].content
print("JUST CONTENT OF ANSWER", output2)

# ========================= REACT AGENT CORE ENDS ============================================

