"""Dataset loading utilities.

This module contains the core interfaces and implementations for loading datasets
from various sources, with appropriate error handling and caching.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union

from datasets import Dataset, DatasetDict, disable_caching, enable_caching
from datasets import disable_progress_bar, enable_progress_bar, load_dataset
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from ember.core.exceptions import GatedDatasetAuthenticationError

logger = logging.getLogger(__name__)


class IDatasetLoader(ABC):
    """Base interface for dataset loaders.

    All dataset loaders must implement this interface to provide consistent
    behavior for loading datasets from various sources.
    """

    @abstractmethod
    def load(
        self, *, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        """Load a dataset by name and optional configuration.

        Args:
            dataset_name: The name of the dataset to load.
            config: Optional configuration parameter for the dataset.

        Returns:
            The loaded dataset (either a DatasetDict or Dataset).

        Raises:
            ValueError: If the dataset cannot be found or loaded.
            RuntimeError: If an error occurs during dataset loading.
        """
        pass


class HuggingFaceDatasetLoader(IDatasetLoader):
    """Loader for datasets from the Hugging Face Hub.

    This loader handles dataset retrieval from the Hugging Face Hub, with appropriate
    error handling, caching, and progress reporting.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize the loader with optional cache directory.

        Args:
            cache_dir: Custom cache directory for datasets. If not provided,
                       the default Hugging Face cache location is used.
        """
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "datasets"
            )
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def load(
        self, *, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        """Loads a dataset from the Hugging Face Hub with robust error handling.

        The method first checks for the dataset's existence on the Hub, then proceeds to load it,
        engaging caching mechanisms and progress indicators. Any HTTP or unexpected errors are logged
        and re-raised as RuntimeError.

        Args:
            dataset_name (str): The name of the dataset to load.
            config (Optional[str]): Optional configuration parameter for the dataset.

        Returns:
            Union[DatasetDict, Dataset]: The resulting dataset object.

        Raises:
            ValueError: If the dataset cannot be found on the Hugging Face Hub.
            RuntimeError: If an HTTP error occurs or an unexpected exception is raised during loading.
        """
        logger.info("Checking dataset existence on the Hub: %s", dataset_name)
        
        api: HfApi = HfApi()
        try:
            api.dataset_info(dataset_name)
        except Exception as exc:
            logger.error("Dataset %s not found on the Hub: %s", dataset_name, exc)
            raise ValueError(
                "Dataset '%s' does not exist on the Hub." % dataset_name
            ) from exc

        logger.info("Loading dataset: %s (config: %s)", dataset_name, config)
        try:
            enable_progress_bar()
            enable_caching()
            dataset: Union[DatasetDict, Dataset] = load_dataset(
                path=dataset_name, name=config, cache_dir=self.cache_dir
            )
            logger.info(
                "Successfully loaded dataset: %s (config: %s)", dataset_name, config
            )
            return dataset
        except HTTPError as http_err:
            logger.error(
                "HTTP error while loading dataset %s: %s", dataset_name, http_err
            )
            raise RuntimeError(
                "Failed to download dataset '%s'." % dataset_name
            ) from http_err
        except Exception as exc:
            # Check for authentication error with gated datasets
            if (
                str(exc).find("is a gated dataset") >= 0
                or str(exc).find("You must be authenticated") >= 0
            ):
                logger.error(
                    "Authentication required for gated dataset %s", dataset_name
                )
                raise GatedDatasetAuthenticationError.for_huggingface_dataset(
                    dataset_name
                ) from exc
            else:
                logger.error(
                    "Unexpected error loading dataset %s: %s", dataset_name, exc
                )
                raise RuntimeError(
                    "Error loading dataset '%s': %s" % (dataset_name, exc)
                ) from exc
        finally:
            disable_caching()
            disable_progress_bar()