import streamlit as st
import time
from agentic_model import run_model


def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ai_typing" not in st.session_state:
        # could be good to have a typing indicator for the AI
        st.session_state.ai_typing = False

    # header
    st.title("Welcome to My Agentic AI Assistant")
    st.write("Enter a prompt that can be accomplished with code and the system will generate the code and run it to provide you an answer.")
    st.write("ex. what is the current stock price of GOOGL?")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = run_model(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.summary)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response.summary})


if __name__ == "__main__":
    main()
