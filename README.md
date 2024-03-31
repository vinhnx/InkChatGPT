<div align="center">
  <img alt="app icon" height="196px" src="./assets/icon.jpg">
</div>

# 📚 [InkChatGPT](https://inkchatgpt.streamlit.app) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inkchatgpt.streamlit.app)

`InkChatGPT` is a `Streamlit` application that allows users to upload PDF documents and engage in a conversational Q&A with a language model (`LLM`) based on the content of those documents.

## Features

-   Upload any PDF documents and start asking key information about it
-   Conversational Q&A with LLM (powered by `OpenAI`'s GPT-3.5-turbo model)
-   Use `HuggingFace` embeddings to generate embeddings for the document chunks with `all-MiniLM-L6-v2` model.
-   Clear conversation history
-   Responsive UI with loading indicators and chat interface

## Prerequisites

-   Python 3.7 or later
-   OpenAI API key (set as an environment variable: `OPENAI_API_KEY`)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/your-username/InkChatGPT.git
cd InkChatGPT
```

2. Create a virtual environment and activate it:

```sh
python -m venv env
source env/bin/activate # On Windows, use env\Scripts\activate
```

3. Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Set the `OPENAI_API_KEY` environment variable with your OpenAI API key:
   export OPENAI_API_KEY=YOUR_API_KEY

2. Run the Streamlit app:

```sh
streamlit run app.py
```

3. Upload PDF documents and start chatting with the LLM!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
