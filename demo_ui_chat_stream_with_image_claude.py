# streamlit run demo_ui_chat_stream_with_image.py

import streamlit as st
import base64, json
from src.chat import Chat
from src.llm_bedrock_claude import ClaudeLLM

default_system_prompt = "Talk like a pirate."

# Create a chat object with the default system prompt
if 'chat' not in st.session_state:
    st.session_state['chat'] = Chat(llm=ClaudeLLM(), system_prompt=default_system_prompt)

# Function to change the system prompt
def change_system_prompt():
    st.session_state['chat'].system_prompt = system_prompt_input

# Display chat messages
messages = st.session_state.chat.messages.dict()['messages']
for message in messages:
    if message["role"] == "assistant":
        avatar="./img/claude.png"
    else:
        avatar=None

    with st.chat_message(message["role"], avatar=avatar):
        for content in message["content"]:

            if "text" == content['type']:
                st.markdown(content["text"])

            elif "image" == content['type']:
                image_data = content["source"]["data"]
                st.image(base64.b64decode(image_data))
                break


# with st.popover("Open popover"):
with st.sidebar:

    # Set a title for the page
    st.title("Claude Chat with Image")

    # Expander box with controls to change the system prompt
    with st.expander("System Prompt", expanded=False):
        system_prompt_input = st.text_area(
            "Enter a system prompt:", 
            value=default_system_prompt,
            on_change=change_system_prompt
        )
        st.markdown("**Try**: `Talk like a pirate.` or `Talk like a tree.` or something much longer.")
        st.button("Save", on_click=change_system_prompt())

    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

    st.markdown("1. Upload the image (_optional_).\n1. Write and send prompt.\n1. Clear the image upload, else it will send again.")

    prompt = st.chat_input("What is up?")

# Accept user input
if prompt:

    message = {
        "role" : "user",
        "content" : [{
            "type": "text",
            "text": prompt
        }]
    }

    # Check if an image was uploaded
    image_base64 = ""
    if uploaded_file is not None:
        # Convert to base64
        mime_type = uploaded_file.type
        bytes_data = uploaded_file.getvalue()
        image_base64 = base64.b64encode(bytes_data).decode()

        message['content'].append({
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_base64
            }
        })

    with st.chat_message("user"):
        st.markdown(prompt)
        if image_base64:
            st.image(base64.b64decode(image_base64))

    # Generate and display response
    with st.chat_message("assistant", avatar="./img/claude.png"):
        response = st.write_stream(st.session_state['chat'].generate_stream(message))

