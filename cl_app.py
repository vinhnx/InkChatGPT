import os

import chainlit as cl
import dotenv
from chromadb.config import Settings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from calback_handler import StreamHandler

# Constants
GPT_LLM_MODEL = "gpt-3.5-turbo"
COMMAND_R_LLM_MODEL = "command-r"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# TODO: USE LLAMA 3 with Local or Grog
# >> TODO checking streaming with improve RAG workflow https://github.com/Chainlit/chainlit/issues/764!
# example https://github.com/Chainlit/cookbook/blob/main/openai-functions-streaming/app.py


dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


async def load_vector_store():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please select files",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
            author="InkChatGPT",
        ).send()

    file = files[0]

    # show processing message
    message = cl.Message(content=f"Processing {file.name}...", disable_feedback=True)
    await message.send()

    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)

    # create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # create a Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = await cl.make_async(Chroma.from_texts)(
        texts,
        embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db",
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )

    return vector_db


def retrieval_qa_chain(vectorstore):
    # Define prompt template
    template = (
        "Combine the chat history and follow up question into "
        "a standalone question. Chat History: {history}"
        "Follow up question: {question}"
    )
    prompt = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="question",
    )

    llm = ChatOpenAI(
        model=GPT_LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        streaming=True,
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    docs_retriever = vectorstore.as_retriever()

    document_variable_name = "context"
    # The prompt here should take as an input variable the
    # `document_variable_name`
    combine_docs_chain_prompt = PromptTemplate.from_template(
        "Summarize this content: {context}"
    )

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name=document_variable_name,
        document_prompt=combine_docs_chain_prompt,
    )

    chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        question_generator=llm_chain,
        retriever=docs_retriever,
        memory=memory,
        verbose=True,
        max_tokens_limit=4000,
        response_if_no_docs_found="I'm sorry, I can't assist with that.",
    )
    return chain


async def make_chain():
    vector_store = await load_vector_store()
    chat = retrieval_qa_chain(vector_store)
    return chat


@cl.on_chat_start
async def start():
    chain = await make_chain()
    msg = cl.Message(content="Firing up the InkChatGPT bot...")

    await msg.send()
    msg.content = "Hi, I'm InkChatGPT. Feel free to ask me any question."
    await msg.update()
    cl.user_session.set("chain", chain)

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    chain = cl.user_session.get("chain")
    await chain.acall(
        message.content,
        callbacks=[
            cl.AsyncLangchainCallbackHandler(),
            StreamHandler(),
        ],
    )
