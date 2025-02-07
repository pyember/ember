from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDiscoveryProvider(ABC):
    @abstractmethod
    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch a dictionary mapping canonical model IDs (e.g. 'openai:gpt-4o')
        to metadata retrieved from the provider's API.
        """
        pass
