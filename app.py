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

# Function to process user input and stream model response
def process_input_stream(user_input):
    # Create chain
    chain = prompt | model

    # Stream the response chunks
    result = chain.stream({"question": user_input})

    # Display the streamed response chunk by chunk
    full_response = ""
    for chunk in result:
        full_response += chunk  # Append each chunk to the full response
        yield full_response  # Yield the response incrementally for streaming effect

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

    # Placeholder for AI response
    response_placeholder = st.empty()

    # Show spinner while AI generates the response
    with st.spinner('Code Llama is thinking...'):
        # Stream and display the AI response incrementally
        full_response = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for chunk in process_input_stream(user_input):
                full_response = chunk
                message_placeholder.markdown(f"{full_response}▌")  # Display with a cursor effect

    # Once streaming is done, finalize the message
    message_placeholder.markdown(f"{full_response}")

    # Append the full AI response to the session state
    st.session_state['messages'].append({"role": "assistant", "content": f"{full_response}"})
