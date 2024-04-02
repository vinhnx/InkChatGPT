import os
import tempfile

import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource(ttl="1h")
def configure_retriever(files, use_compression=False):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        _, extension = os.path.splitext(temp_filepath)

        # Load the file using the appropriate loader
        if extension == ".pdf":
            loader = PyPDFLoader(temp_filepath)
        elif extension == ".docx":
            loader = Docx2txtLoader(temp_filepath)
        elif extension == ".txt":
            loader = TextLoader(temp_filepath)
        else:
            st.write("This document format is not supported!")
            return None

        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    if not use_compression:
        return retriever

    embeddings_filter = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.76
    )

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=retriever
    )
