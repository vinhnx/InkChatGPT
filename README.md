<p align="center">
  <img src="./assets/large_icon.png" height="200" alt="icon" />
</p>

<p align="center">
  <em>📚 InkChatGPT - Chat with Documents</em>
</p>

<p align="center">
   <a href="https://inkchatgpt.streamlit.app/"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a>
</p>

<p align="center">
<b><a href="https://x.com/vinhnx">Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://github.com/vinhnx">GitHub</a></b>
</p>

# InkChatGPT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![GitHub User's stars](https://img.shields.io/github/stars/vinhnx)](https://github.com/vinhnx)
[![HackerNews User Karma](https://img.shields.io/hackernews/user-karma/vinhnx)](https://news.ycombinator.com/user?id=vinhnx)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/vinhnx)](https://x.com/vinhnx)

`InkChatGPT` is a `Streamlit` application that allows users to upload PDF documents and engage in a conversational Q&A with a language model (`LLM`) based on the content of those documents.

### Features

-   Upload any documents and start asking key information about it, currently supports: PDF, TXT, DOCX, EPUB
-   Limit 200MB per file
-   Conversational Q&A with LLM (powered by `OpenAI`'s `gpt-3.5-turbo` model)
-   `HuggingFace` embeddings to generate embeddings for the document chunks with `all-MiniLM-L6-v2` model.
-   `VectorDB` for document vector retrieval storage

## Prerequisites

-   Python 3.7 or later
-   OpenAI API key (set as an environment variable: `OPENAI_API_KEY`)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/vinhnx/InkChatGPT.git
cd InkChatGPT
```

2. Create a virtual environment and activate it:

```sh
python -m venv env
source env/bin/activate
```

3. Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

1. Set the `OPENAI_API_KEY` environment variable with your OpenAI API key:

```sh
export OPENAI_API_KEY=YOUR_API_KEY
```

2. Run the Streamlit app:

```sh
streamlit run app.py
```

3. Upload PDF documents and start chatting with the LLM!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).
