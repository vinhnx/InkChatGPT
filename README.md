<div align="center">
  <img alt="app icon" height="196px" src="./assets/icon.jpg">
</div>

# [InkChatGPT](https://inkchatgpt.streamlit.app)

InkChatGPT is a Simple LLM app that demonstrates a Retrieval-Augmented Generation (RAG) model for question-answering using LangChain, ChromaDB, and OpenAI's language models.

The app allows users to upload documents (PDFs or text files), and then ask questions related to the content of those documents. The RAG model retrieves relevant passages from the documents and generates an answer based on the retrieved context.

👉 Run the app, online: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inkchatgpt.streamlit.app)

---

Running RAG to ask question about certain information inside the Document.

> [!NOTE]  
> For this demo, I use asked the model about the [How to Build Your Career in AI](https://info.deeplearning.ai/how-to-build-a-career-in-ai-book) PDF from Andrew Ng.

![Demo Screenshot](./assets/demo.jpg)

---

## Features

-   **Document Upload**: Supports uploading PDF and text files.
-   **Question Answering**: Ask questions related to the uploaded documents.
-   **Retrieval-Augmented Generation (RAG)**: Utilizes a RAG model for context-aware question answering.
-   **LangChain Integration**: Leverages LangChain for building the RAG model.
-   **ChromaDB**: Efficient vector storage and retrieval using ChromaDB.
-   **OpenAI Language Models**: Use `gpt-3.5-turbo` embedding model for answer generation.
-   **Streamlit UI**: Interactive and user-friendly web interface.

## Local Installation

1. Clone the repository:

```shellscript
git clone https://github.com/vinhnx/InkChatGPT.git
```

2. Navigate to the project directory:

```shellscript
cd InkChatGPT
```

3. Create a new Python environment and activate it (e.g., using `venv` or `conda`).

4. Install the required packages:

```shellscript
pip install streamlit langchain chromadb openai tiktoken pypdf
```

or, run

```shellscript
pip install -r requirements.txt
```

## Usage

1. **Open VSCode and Create a New Python Environment**:

-   Open Visual Studio Code.
-   Open the Command Palette by pressing `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
-   Search for "Python: Create Environment" and select it.
-   Choose the option to create a new virtual environment.
-   Give the environment a name (e.g., "llm-rag-env") and select the Python interpreter version you want to use.

2. **Select the Python Interpreter**:

-   Once the virtual environment is created, you'll be prompted to select the Python interpreter.
-   From the Command Palette, search for "Python: Select Interpreter" and choose the interpreter you just created (e.g., "llm-rag-env").

3. **Open the Project Folder**:

-   From the File menu, choose "Open Folder" or "Open..." and navigate to the project folder containing the `app.py` file.

4. **Configure the App**:

-   Navigate to OpenAI Platform > API Keys to generate an API Key to run the model
-   Copy the API KEY (start with sk-)

5. **Run the Streamlit App**:

-   In the terminal, navigate to the project directory if you're not already there.
-   Run the following command to start the Streamlit app:
    ```shellscript
    streamlit run app.py
    ```
-   This will start the Streamlit app and provide you with a local URL to access the app in your web browser.
-   Pass your OpenAI API key into the `OpenAI API Key` field

6. **Use the App**:

-   Open the provided local URL in your web browser.
-   You should see the InkChatGPT interface.
-   Follow the instructions in the app to upload documents and ask questions.

## Contributing

Contributions are welcome! If you'd like to improve the InkChatGPT, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Create a new Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

I'm Vinh, [@vinhnx](https://x.com/vinhnx) on almost everywhere. Feel free to reach out with any questions or feedback!
