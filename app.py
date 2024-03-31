import os
import streamlit as st

from chat_profile import ChatProfileRoleEnum
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# config page
st.set_page_config(page_title="InkChatGPT", page_icon="📚")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")


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
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.OPENAI_API_KEY)
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def main():
    """
    The main function that runs the Streamlit app.
    """

    if st.secrets.OPENAI_API_KEY:
        openai_api_key = st.secrets.OPENAI_API_KEY
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        st.secrets.OPENAI_API_KEY = openai_api_key

        if not st.secrets.OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")

    if len(msgs.messages) == 0:
        msgs.add_ai_message(
            """
            Hello, how can I help you?

            You can upload a document and chat with me to ask questions related to its content.
        """
        )

    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    if question := st.chat_input(
        placeholder="Chat with your document",
        disabled=(not openai_api_key),
    ):
        st.chat_message(ChatProfileRoleEnum.Human).write(question)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI chatbot having a conversation with a human."),
                MessagesPlaceholder(variable_name="history"),
                (ChatProfileRoleEnum.Human, f"{question}"),
            ]
        )

        llm = ChatOpenAI(
            openai_api_key=st.secrets.OPENAI_API_KEY,
            temperature=0.0,
            model_name="gpt-3.5-turbo",
        )

        chain = prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Note: new messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": question}, config)
        st.chat_message(ChatProfileRoleEnum.AI).write(response.content)


def build_sidebar():
    with st.sidebar:
        st.title("📚 InkChatGPT")
        uploaded_file = st.file_uploader(
            "Select a file", type=["pdf", "docx", "txt"], key="file_uploader"
        )

        add_file = st.button(
            "Process File",
            disabled=(not uploaded_file and not st.secrets.OPENAI_API_KEY),
        )
        if add_file and uploaded_file and st.secrets.OPENAI_API_KEY.startswith("sk-"):
            with st.spinner("💭 Thinking..."):
                vector_store = load_and_process_file(uploaded_file)

                if vector_store:
                    msgs.add_ai_message(
                        f"""
                        File: `{uploaded_file.name}`, processed successfully!

                        Feel free to ask me any question.
                        """
                    )


if __name__ == "__main__":
    build_sidebar()
    main()
