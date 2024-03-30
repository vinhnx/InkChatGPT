import os

from dotenv import load_dotenv

load_dotenv()

llm_api_key = os.environ.get("OPENAI_API_KEY")
