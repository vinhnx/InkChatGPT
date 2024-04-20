import chainlit as cl

from langchain.indexes import SQLRecordManager, index
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableConfig, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from calback_handler import PostMessageHandler

# Constants
GPT_LLM_MODEL = "gpt-3.5-turbo"


async def process_pdfs():
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    files = []
    # while files is None:
    files = await cl.AskFileMessage(
        content="Please select PDF files",
        accept=["application/pdf"],
        max_size_mb=20,
        timeout=180,
    ).send()

    file = files[0]

    # show processing message
    message = cl.Message(
        content="Processing file. Please wait a moment...",
        disable_feedback=True,
    )
    await message.send()

    loader = PyMuPDFLoader(str(file.path))
    documents = loader.load()
    docs += text_splitter.split_documents(documents)

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search


@cl.on_chat_start
async def on_chat_start():
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    doc_search = await process_pdfs()
    retriever = doc_search.as_retriever()

    model = ChatOpenAI(
        model=GPT_LLM_MODEL,
        streaming=True,
        temperature=0,
    )

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    message = cl.Message(content="Firing up the InkChatGPT bot...")
    message.content = "Hi, I'm InkChatGPT. Feel free to ask me any question."
    await message.send()

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

    await msg.send()
