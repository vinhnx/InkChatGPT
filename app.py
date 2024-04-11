import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_community.chat_models.openai import ChatOpenAI

from calback_handler import PrintRetrievalHandler, StreamHandler
from chat_profile import ChatProfileRoleEnum
from document_retriever import configure_retriever

LLM_MODEL = "gpt-3.5-turbo"

st.set_page_config(
    page_title="InkChatGPT: Chat with Documents",
    page_icon="📚",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://x.com/vinhnx",
        "Report a bug": "https://github.com/vinhnx/InkChatGPT/issues",
        "About": """InkChatGPT is a simple Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a conversational Q&A, with a language model (LLM) based on the content of those documents.
        
        GitHub: https://github.com/vinhnx/InkChatGPT""",
    },
)

# Hide Header
# st.markdown(
#     """<style>.stApp [data-testid="stToolbar"]{display:none;}</style>""",
#     unsafe_allow_html=True,
# )

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()

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
        st.caption(
            """
            Simple Retrieval Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a conversational Q&A, with a language model (LLM) based on the content of those documents. Built with LangChain as Streamlit.
            
            Supports PDF, TXT, DOCX • Limit 200MB per file.
            * GitHub: https://github.com/vinhnx/InkChatGPT
            * Twitter: https://x.com/vinhnx
            """
        )

chat_tab, documents_tab, settings_tab = st.tabs(["Chat", "Documents", "Settings"])
with settings_tab:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if len(msgs.messages) == 0 or st.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("""
        Hi, your uploaded document(s) had been analyzed. 
        
        Feel free to ask me any questions. For example: you can start by asking me `'What is this book about?` or `Tell me about the content of this book!`' 
        """)

with documents_tab:
    uploaded_files = st.file_uploader(
        label="Select files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        disabled=(not openai_api_key),
    )

with chat_tab:
    if uploaded_files:
        result_retriever = configure_retriever(uploaded_files)

        if result_retriever is not None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                chat_memory=msgs,
                return_messages=True,
            )

            # Setup LLM and QA chain
            llm = ChatOpenAI(
                model=LLM_MODEL,
                api_key=openai_api_key,
                temperature=0,
                streaming=True,
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=result_retriever,
                memory=memory,
                verbose=False,
                max_tokens_limit=4000,
            )

            avatars = {
                ChatProfileRoleEnum.HUMAN: "user",
                ChatProfileRoleEnum.AI: "assistant",
            }
            
            for msg in msgs.messages:
                st.chat_message(avatars[msg.type]).write(msg.content)

if not openai_api_key:
    st.caption("🔑 Add your **OpenAI API key** on the `Settings` to continue.")

if user_query := st.chat_input(
    placeholder="Ask me anything!",
    disabled=(not openai_api_key),
):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.empty())
        stream_handler = StreamHandler(st.empty())
        response = chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
