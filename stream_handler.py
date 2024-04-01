import os
import streamlit as st

from langchain.callbacks.base import BaseCallbackHandler


# Callback handlers
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
