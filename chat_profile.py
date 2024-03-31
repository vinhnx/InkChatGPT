from langchain.schema import ChatMessage
from enum import Enum


class ChatProfileRoleEnum(str, Enum):
    User = "user"
    Assistant = "assistant"


class ChatProfile:
    def __init__(self, role: str, message: str):
        self.role = role
        self.message = message

    def build_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.message)


class Assistant(ChatProfile):
    def __init__(self, message: str):
        super().__init__(ChatProfileRoleEnum.Assistant, message)


class User(ChatProfile):
    def __init__(self, message: str):
        super().__init__(ChatProfileRoleEnum.User, message)
