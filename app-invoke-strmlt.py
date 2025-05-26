import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from decouple import config
import os

# Enable tracing and API keys
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = config("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = config("LANGCHAIN_PROJECT")

# Set up the template and model
template = """Question: {question}

Answer: (in one sentence please...)"""
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="codellama")

# Initialize session state if not present
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to process user input and get model response
def process_input(user_input):
    response = model.invoke(user_input)  # Get response as a string
    return response

# Streamlit UI
st.title("Code Llama w/ Ollama - a local code assistant 🦙")

# Display the conversation history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Input field at the bottom
user_input = st.chat_input("Ask your question about coding:")

# When user submits input
if user_input:
    # Append user's input to messages and display it immediately
    st.session_state['messages'].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Show spinner while AI generates the response
    with st.spinner('Code Llama is thinking...'):
        # Process input and get AI response
        ai_response = process_input(user_input)

    # Display the AI response and update the conversation
    with st.chat_message("assistant"):
        st.markdown(f"{ai_response}")

    st.session_state['messages'].append({"role": "assistant", "content": ai_response})
