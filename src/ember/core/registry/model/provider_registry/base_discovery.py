from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDiscoveryProvider(ABC):
    """Abstract base class for model discovery providers.

    Implementations must fetch model metadata from their respective APIs, returning
    a mapping from canonical model IDs to detailed metadata.
    """

    @abstractmethod
    def fetch_models(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve model metadata from the provider's API.

        This method must be overridden by subclasses to provide a dictionary mapping
        canonical model IDs (e.g., 'openai:gpt-4o') to their metadata, as obtained from
        the provider's API.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping where each key is a canonical model ID
            and its value is a dictionary containing model metadata.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement fetch_models.")
