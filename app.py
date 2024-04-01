import os
import tempfile

import streamlit as st
from chat_profile import ChatProfileRoleEnum

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from streamlit_extras.add_vertical_space import add_vertical_space

# TODO: refactor
# TODO: extract class
# TODO: modularize
# TODO: hide side bar
# TODO: make the page attactive

# configs
LLM_MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title=":books: InkChatGPT: Chat with Documents",
    page_icon="📚",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://x.com/vinhnx",
        "Report a bug": "https://github.com/vinhnx/InkChatGPT/issues",
        "About": "InkChatGPT is a Streamlit application that allows users to upload PDF documents and engage in a conversational Q&A with a language model (LLM) based on the content of those documents.",
    },
)

st.image("./assets/icon.jpg", width=100)
st.header(
    ":gray[:books: InkChatGPT]",
    divider="blue",
)
st.write("**Chat** with Documents")

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Thinking...**")
        self.container = container

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Checking document for query:** `{query}`. Please wait...")

    def on_retriever_end(self, documents, **kwargs):
        self.container.empty()


with st.sidebar.expander("Documents"):
    st.subheader("Files")
    uploaded_files = st.file_uploader(
        label="Select PDF files", type=["pdf"], accept_multiple_files=True
    )


with st.sidebar.expander("Setup"):
    st.subheader("API Key")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    is_empty_chat_messages = len(msgs.messages) == 0
    if is_empty_chat_messages or st.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

if not openai_api_key:
    st.info("Please add your OpenAI API key in the sidebar to continue.")
    st.stop()

if uploaded_files:
    retriever = configure_retriever(uploaded_files)

    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=msgs, return_messages=True
    )

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        openai_api_key=openai_api_key,
        temperature=0,
        streaming=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=False
    )

    avatars = {
        ChatProfileRoleEnum.Human: "user",
        ChatProfileRoleEnum.AI: "assistant",
    }

    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.empty())
            stream_handler = StreamHandler(st.empty())
            response = chain.run(
                user_query, callbacks=[retrieval_handler, stream_handler]
            )
