import os

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from apikey import llm_api_key

key = llm_api_key


def load_and_process_file(file_data):
    """
    Load and process the uploaded file.
    Returns a vector store containing the embedded chunks of the file.
    """
    file_name = os.path.join("./", file_data.name)
    with open(file_name, "wb") as f:
        f.write(file_data.getvalue())

    _, extension = os.path.splitext(file_name)

    # Load the file using the appropriate loader
    if extension == ".pdf":
        loader = PyPDFLoader(file_name)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_name)
    elif extension == ".txt":
        loader = TextLoader(file_name)
    else:
        st.write("This document format is not supported!")
        return None

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    return vector_store


def initialize_chat_model(vector_store):
    """
    Initialize the chat model with the given vector store.
    Returns a ConversationalRetrievalChain instance.
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=key,
    )
    retriever = vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(llm, retriever)


def main():
    """
    The main function that runs the Streamlit app.
    """
    st.set_page_config(page_title="InkChatGPT", page_icon="📚")

    st.title("📚 InkChatGPT")
    st.write("Upload a document and ask questions related to its content.")

    uploaded_file = st.file_uploader(
        "Select a file", type=["pdf", "docx", "txt"], key="file_uploader"
    )

    if uploaded_file:
        add_file = st.button(
            "Process File",
            on_click=clear_history,
            key="process_button",
        )

    if uploaded_file and add_file:
        with st.spinner("💭 Thinking..."):
            vector_store = load_and_process_file(uploaded_file)
            if vector_store:
                crc = initialize_chat_model(vector_store)
                st.session_state.crc = crc
                st.success("File processed successfully!")

    if "crc" in st.session_state:
        st.markdown("## Ask a Question")
        question = st.text_area(
            "Enter your question",
            height=93,
            key="question_input",
        )

        submit_button = st.button("Submit", key="submit_button")

        if submit_button and "crc" in st.session_state:
            handle_question(question)

        display_chat_history()


def handle_question(question):
    """
    Handles the user's question by generating a response and updating the chat history.
    """
    crc = st.session_state.crc
    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.spinner("Generating response..."):
        response = crc.run(
            {
                "question": question,
                "chat_history": st.session_state["history"],
            }
        )

    st.session_state["history"].append((question, response))
    st.write(response)


def display_chat_history():
    """
    Displays the chat history in the Streamlit app.
    """
    if "history" in st.session_state:
        st.markdown("## Chat History")
        for q, a in st.session_state["history"]:
            st.markdown(f"**Question:** {q}")
            st.write(a)
            st.write("---")


def clear_history():
    """
    Clear the chat history stored in the session state.
    """
    if "history" in st.session_state:
        del st.session_state["history"]


if __name__ == "__main__":
    main()
    main()
