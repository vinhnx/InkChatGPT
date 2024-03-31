import os
import streamlit as st

from token_stream_handler import StreamHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import ChatMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.set_page_config(page_title="InkChatGPT", page_icon="📚")


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
        st.error("This document format is not supported!")
        return None

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
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
        openai_api_key=st.session_state.api_key,
    )
    retriever = vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(llm, retriever)


def main():
    """
    The main function that runs the Streamlit app.
    """

    assistant_message = "Hello, you can upload a document and chat with me to ask questions related to its content."
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content=assistant_message)
    ]

    st.chat_message("assistant").write(assistant_message)

    if prompt := st.chat_input(
        placeholder="Chat with your document",
        disabled=(not st.session_state.api_key),
    ):
        st.session_state.messages.append(
            ChatMessage(
                role="user",
                content=prompt,
            )
        )
        st.chat_message("user").write(prompt)

        handle_question(prompt)


def handle_question(question):
    """
    Handles the user's question by generating a response and updating the chat history.
    """
    crc = st.session_state.crc

    if "history" not in st.session_state:
        st.session_state["history"] = []

    response = crc.run(
        {
            "question": question,
            "chat_history": st.session_state["history"],
        }
    )

    st.session_state["history"].append((question, response))

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(
            openai_api_key=st.session_state.api_key,
            streaming=True,
            callbacks=[stream_handler],
        )
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response.content)
        )


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


def process_data(uploaded_file, openai_api_key):
    if uploaded_file and openai_api_key.startswith("sk-"):
        with st.spinner("💭 Thinking..."):
            vector_store = load_and_process_file(uploaded_file)

            if vector_store:
                crc = initialize_chat_model(vector_store)
                st.session_state.crc = crc
                st.success(f"File: `{uploaded_file.name}`, processed successfully!")


def build_sidebar():
    with st.sidebar:
        st.title("📚 InkChatGPT")

        with st.form(key="input_form"):
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key",
            )

            st.session_state.api_key = openai_api_key
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")

            uploaded_file = st.file_uploader(
                "Select a file", type=["pdf", "docx", "txt"], key="file_uploader"
            )

            st.form_submit_button(
                "Process File",
                on_click=process_data(
                    uploaded_file=uploaded_file, openai_api_key=openai_api_key
                ),
            )


if __name__ == "__main__":
    build_sidebar()
    main()
