from langchain.callbacks.base import BaseCallbackHandler
import chainlit as cl


class PostMessageHandler(BaseCallbackHandler):
    """
    Callback handler for handling the retriever and LLM processes.
    Used to post the sources of the retrieved documents as a Chainlit element.
    """

    def __init__(self, msg: cl.Message):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = set()  # To store unique pairs

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        for d in documents:
            source_page_pair = (d.metadata["source"], d.metadata["page"])
            self.sources.add(source_page_pair)  # Add unique pairs to the set

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            sources_text = "\n".join(
                [f"{source}#page={page}" for source, page in self.sources]
            )
            self.msg.elements.append(
                cl.Text(name="Sources", content=sources_text, display="inline")
            )
