from enum import Enum


class LLMProviderEnum(str, Enum):
    OPEN_AI = "OpenAI"
    COHERE = "Cohere"
