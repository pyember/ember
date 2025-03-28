import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union
from urllib.error import HTTPError

from datasets import Dataset, DatasetDict, disable_caching, enable_caching, load_dataset
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from ember.core.exceptions import GatedDatasetAuthenticationError
from huggingface_hub import HfApi

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IDatasetLoader(ABC):
    """Interface for dataset loaders.

    Subclasses must implement the load method to retrieve a dataset given a name and an optional configuration.
    """

    @abstractmethod
    def load(
        self, *, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        """Loads a dataset from a specified source.

        Args:
            dataset_name (str): The identifier of the dataset to load.
            config (Optional[str]): An optional configuration name for the dataset.

        Returns:
            Union[DatasetDict, Dataset]: The loaded dataset.

        Raises:
            Exception: Subclasses should raise an appropriate Exception if loading fails.
        """
        pass


class HuggingFaceDatasetLoader(IDatasetLoader):
    """Loads datasets from the Hugging Face Hub.

    This class verifies the dataset's existence on the Hub before loading it,
    utilizing caching and progress bar controls to enhance the user experience.
    """

    def __init__(self, *, cache_dir: Optional[str] = None) -> None:
        """Initializes the Hugging Face dataset loader.

        Args:
            cache_dir (Optional[str]): Custom directory path for caching datasets.
                If not provided, defaults to "~/.cache/huggingface/datasets".
        """
        self.cache_dir: str = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "datasets"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

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
