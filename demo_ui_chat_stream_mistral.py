# streamlit run demo_ui_chat_stream.py

import streamlit as st
import base64
from src.chat import Chat
from src.llm_bedrock_mistral import MistralLLM

default_system_prompt = """Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. 
You are a friendly and helpful chatbot. 
You will answer any questions you are asked to the best of your ability, 
and you MUST do so in the voice of a pirate."""

# Set a title for the page
st.title("Mistral Chat")

# Create a chat object
if 'chat' not in st.session_state:
    st.session_state['chat'] = Chat(llm=MistralLLM(), system_prompt=default_system_prompt)

# Function allows you to change the system prompt
def change_system_prompt():
    st.session_state['chat'].system_prompt = system_prompt_input

# expander box with controls to change the system prompt
with st.expander("System Prompt", expanded=False):
    system_prompt_input = st.text_area(
        "Enter a system prompt:", 
        value=default_system_prompt,
        on_change=change_system_prompt
    )
    st.markdown("**Try**: `Talk like a pirate.` or `Talk like a tree.` or something much longer.")
    st.button("Save", on_click=change_system_prompt())

# Display chat messages from history on app rerun
messages = st.session_state.chat.messages.dict()['messages']
for message in messages:

    # Set the avatar depending on who is talking. 
    if message["role"] == "assistant":
        avatar="./img/mistral.png"
    else:
        avatar=None

    with st.chat_message(message["role"], avatar=avatar):
        for content in message["content"]:
            # handle a text message
            if "text" in content:
                st.markdown(message["content"][0]["text"])
            # handle an image
            if "source" in content:
                image_data = message["content"][0]["source"]["data"]
                st.image(base64.b64decode(image_data))
                break

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="./img/mistral.png"):
        response = st.write_stream(st.session_state['chat'].generate_stream(prompt))