import streamlit as st
import requests
import openai
import datetime
from pytz import timezone

# Load system prompts
prompts = {
    "Novice (JLPT N5)": open("system_prompts/novice.txt").read(),
    "Intermediate (JLPT N4-N3)": open("system_prompts/intermediate.txt").read()
}

# Configure OpenRouter API
openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
openrouter_url = "https://openrouter.ai/api/v1/"
client = openai.OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)

st.title("Japanese Assistant")

# Function to generate formatted conversation text
def get_formatted_conversation(level):
    conversation_text = f"Japanese Assistant Conversation ({level})\n"
    conversation_text += f"Exported: {datetime.datetime.now(timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            conversation_text += f"[{msg['role'].upper()}]\n{msg['content']}\n\n"
    
    return conversation_text

with st.sidebar:
    st.header("Settings")
    level = st.radio(
        "Select proficiency level:",
        options=list(prompts.keys()),
        index=0
    )
    system_prompt = st.text_area("Prompt Text", value=prompts[level])

    if "show_export" not in st.session_state or st.session_state.show_export == False:
        if st.button("Show Exportable Text") and "messages" in st.session_state:
            st.session_state.show_export = True
            st.rerun()
    else:
        if st.button("Close Export View"):
            st.session_state.show_export = False
            st.rerun()
    
    if st.button("Clear Conversation"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        if "show_export" in st.session_state:
            st.session_state.show_export = False
        st.rerun()

# Show exportable text if requested
if "show_export" in st.session_state and st.session_state.show_export:
    st.sidebar.subheader("Conversation Export")
    conversation_text = get_formatted_conversation(level)
    st.sidebar.text_area("Copy this text:", conversation_text, height=300)

# Initialize session state
if "messages" not in st.session_state or "current_level" not in st.session_state:
    st.session_state.current_level = level
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
elif st.session_state.current_level != level:
    # Update system message with new level prompt
    st.session_state.current_level = level
    st.session_state.messages[0] = {"role": "system", "content": system_prompt}

# Display chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input text using chat input
user_input = st.chat_input("Ask something in Japanese or English...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
        
    # Get assistant response
    with st.chat_message("assistant"):
        payload = {
            "model": "openai/gpt-4o",
            "messages": st.session_state.messages,
            "stream": True
        }
        
        stream = client.chat.completions.create(**payload)
        response = st.write_stream(stream)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
                