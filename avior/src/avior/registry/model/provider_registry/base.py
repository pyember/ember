import abc
from typing import Any
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.chat_schemas import ChatRequest, ChatResponse


class BaseProviderModel(abc.ABC):
    """
    Abstract base class that all provider implementations must extend.
    """

    def __init__(self, model_info: ModelInfo) -> None:
        self.model_info = model_info
        self.client = self.create_client()

    @abc.abstractmethod
    def create_client(self) -> Any:
        """
        Initialize or configure the API client.
        """
        pass

    @abc.abstractmethod
    def forward(self, request: ChatRequest) -> ChatResponse:
        """
        Given a ChatRequest, invoke the provider and return a ChatResponse.
        """
        pass

    def __call__(self, prompt: str, **kwargs) -> ChatResponse:
        """
        Syntactic sugar that allows your code to do `model("some prompt")`.
        """
        chat_request = ChatRequest(prompt=prompt, **kwargs)
        return self.forward(chat_request)
