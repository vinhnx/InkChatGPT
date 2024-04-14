from sklearn import model_selection
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_cohere import ChatCohere
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_openai import ChatOpenAI

from calback_handler import PrintRetrievalHandler, StreamHandler
from chat_profile import ChatProfileRoleEnum
from document_retriever import configure_retriever
from llm_provider import LLMProviderEnum

# Constants
GPT_LLM_MODEL = "gpt-3.5-turbo"
COMMAND_R_LLM_MODEL = "command-r"

# Properties
uploaded_files = []
api_key = ""
result_retriever = None
chain = None
llm = None
model_name = ""

# Set up sidebar
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# Streamlit app configuration
st.set_page_config(
    page_title="InkChatGPT: Chat with Documents",
    page_icon="📚",
    initial_sidebar_state=st.session_state.sidebar_state,
    menu_items={
        "Get Help": "https://x.com/vinhnx",
        "Report a bug": "https://github.com/vinhnx/InkChatGPT/issues",
        "About": """InkChatGPT is a simple Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a conversational Q&A, with a language model (LLM) based on the content of those documents.
        
        GitHub: https://github.com/vinhnx/InkChatGPT""",
    },
)

with st.sidebar:
    with st.container():
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.image(
                "./assets/app_icon.png",
                use_column_width="always",
                output_format="PNG",
            )
        with col2:
            st.header(":books: InkChatGPT")

        # Model
        selected_model = st.selectbox(
            "Select a model",
            options=[
                LLMProviderEnum.OPEN_AI.value,
                LLMProviderEnum.COHERE.value,
            ],
            index=None,
            placeholder="Select a model...",
        )

        if selected_model:
            api_key = st.text_input(f"{selected_model} API Key", type="password")
            if selected_model == LLMProviderEnum.OPEN_AI:
                model_name = GPT_LLM_MODEL
            elif selected_model == LLMProviderEnum.COHERE:
                model_name = COMMAND_R_LLM_MODEL

        msgs = StreamlitChatMessageHistory()
        if len(msgs.messages) == 0:
            msgs.clear()
            msgs.add_ai_message("""
            Hi, your uploaded document(s) had been analyzed. 
            
            Feel free to ask me any questions. For example: you can start by asking me something like: 
            
            `What is this context about?`
            
            `Help me summarize this!`
            """)

        if api_key:
            # Documents
            uploaded_files = st.file_uploader(
                label="Select files",
                type=["pdf", "txt", "docx"],
                accept_multiple_files=True,
                disabled=(not selected_model),
            )

if api_key and not uploaded_files:
    st.info("🌟 You can upload some documents to get started")

# Check if a model is selected
if not selected_model:
    st.info(
        "📺 Please select a model first, open the `Settings` tab from side bar menu to get started"
    )

# Check if API key is provided
if selected_model and len(api_key.strip()) == 0:
    st.warning(
        f"🔑 API key for {selected_model} is missing or invalid. Please provide a valid API key."
    )

# Process uploaded files
if uploaded_files:
    result_retriever = configure_retriever(uploaded_files, cohere_api_key=api_key)

    if result_retriever is not None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=msgs,
            return_messages=True,
        )

        if selected_model == LLMProviderEnum.OPEN_AI:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0,
                streaming=True,
            )
        elif selected_model == LLMProviderEnum.COHERE:
            llm = ChatCohere(
                model=model_name,
                temperature=0.3,
                streaming=True,
                cohere_api_key=api_key,
            )

        if llm is None:
            st.error(
                "Failed to initialize the language model. Please check your configuration."
            )

        # Create the ConversationalRetrievalChain instance using the llm instance
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=result_retriever,
            memory=memory,
            verbose=True,
            max_tokens_limit=4000,
        )

        avatars = {
            ChatProfileRoleEnum.HUMAN.value: "user",
            ChatProfileRoleEnum.AI.value: "assistant",
        }

        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

# Get user input and generate response
if user_query := st.chat_input(
    placeholder="Ask me anything!",
    disabled=(not uploaded_files),
):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.empty())
        stream_handler = StreamHandler(st.empty())
        response = chain.run(
            user_query,
            callbacks=[retrieval_handler, stream_handler],
        )

if selected_model and model_name:
    st.sidebar.caption(f"🪄 Using `{model_name}` model")
