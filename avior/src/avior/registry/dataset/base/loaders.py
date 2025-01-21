import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
from datasets import Dataset, DatasetDict, load_dataset
from datasets.utils.logging import enable_progress_bar, disable_progress_bar
from datasets import enable_caching, disable_caching
from huggingface_hub import HfApi
from urllib.error import HTTPError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IDatasetLoader(ABC):
    @abstractmethod
    def load(
        self, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        pass


class HuggingFaceDatasetLoader(IDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "datasets"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def load(
        self, dataset_name: str, config: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        logger.info(f"Checking dataset existence on the Hub: {dataset_name}")
        api = HfApi()
        try:
            api.dataset_info(dataset_name)
        except Exception as e:
            logger.error(f"Dataset {dataset_name} not found on the Hub: {e}")
            raise ValueError(
                f"Dataset '{dataset_name}' does not exist on the Hub."
            ) from e

        logger.info(f"Loading dataset: {dataset_name} (config: {config})")
        try:
            enable_progress_bar()
            enable_caching()
            dataset = load_dataset(dataset_name, config, cache_dir=self.cache_dir)
            logger.info(
                f"Successfully loaded dataset: {dataset_name} (config: {config})"
            )
            return dataset
        except HTTPError as e:
            logger.error(f"HTTP error while loading dataset {dataset_name}: {e}")
            raise RuntimeError(f"Failed to download dataset '{dataset_name}'.") from e
        except Exception as e:
            logger.error(f"Unexpected error loading dataset {dataset_name}: {e}")
            raise RuntimeError(f"Error loading dataset '{dataset_name}': {e}") from e
        finally:
            disable_caching()
            disable_progress_bar()
