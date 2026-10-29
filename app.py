import asyncio
import streamlit as st
from orchestrator import orchestrate_question

st.set_page_config(page_title="Shopping Assistant", page_icon="🛒")
st.title("🛒 Shopping Assistant")

# Initialize session states
if "orchestrator_messages" not in st.session_state:
    st.session_state.orchestrator_messages = None

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# Render chat history
for turn in st.session_state.chat_display:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

# Chat input (must be at root level)
if user_question := st.chat_input("Ask the assistant..."):
    # Display user message
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.chat_display.append({"role": "user", "content": user_question})

    # Run orchestrator & display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, st.session_state.orchestrator_messages = asyncio.run(
                orchestrate_question(user_question, st.session_state.orchestrator_messages)
            )
        st.write(answer)
    st.session_state.chat_display.append({"role": "assistant", "content": answer})