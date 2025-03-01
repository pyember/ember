import logging
import random
from typing import Any, Iterable, List, Optional, Union

from ember.core.utils.data.base.loaders import IDatasetLoader
from ember.core.utils.data.base.validators import IDatasetValidator
from ember.core.utils.data.base.samplers import IDatasetSampler
from ember.core.utils.data.base.transformers import IDatasetTransformer
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.models import DatasetEntry, DatasetInfo
from ember.core.utils.data.base.config import BaseDatasetConfig

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatasetService:
    """Service for orchestrating operations on datasets such as loading, validating,
    transforming, sampling, and preparing dataset entries.

    The pipeline executed by this service follows these sequential steps:
      1. Load the dataset from a given source.
      2. Validate the overall structure of the dataset.
      3. Optionally select a specific split.
      4. Apply sequential transformations.
      5. Validate the presence of required keys.
      6. Downsample the dataset if desired.
      7. Prepare the final dataset entries.
    """

    def __init__(
        self,
        loader: IDatasetLoader,
        validator: IDatasetValidator,
        sampler: IDatasetSampler,
        transformers: Optional[Iterable[IDatasetTransformer]] = None,
    ) -> None:
        """Initialize a DatasetService instance.

        Args:
            loader (IDatasetLoader): An instance responsible for loading datasets.
            validator (IDatasetValidator): An instance responsible for validating dataset structures.
            sampler (IDatasetSampler): An instance for sampling dataset records.
            transformers (Optional[Iterable[IDatasetTransformer]]): An optional iterable of transformers
                applied sequentially to the dataset.
        """
        self._loader: IDatasetLoader = loader
        self._validator: IDatasetValidator = validator
        self._sampler: IDatasetSampler = sampler
        self._transformers: List[IDatasetTransformer] = (
            list(transformers) if transformers else []
        )

    def _resolve_config_name(
        self, config: Union[str, BaseDatasetConfig, None]
    ) -> Optional[str]:
        """Convert a configuration parameter into a string compatible with the loader.

        Args:
            config (Union[str, BaseDatasetConfig, None]): A configuration identifier provided as a string,
                a BaseDatasetConfig instance, or None.

        Returns:
            Optional[str]: A configuration string if resolvable; otherwise, None.
        """
        if isinstance(config, BaseDatasetConfig):
            return getattr(config, "config_name", None)
        if isinstance(config, str):
            return config
        return None

    def _load_data(self, dataset_name: str, config: Optional[str] = None) -> Any:
        """Load data from a specified dataset using an optional configuration string.

        Args:
            dataset_name (str): The identifier of the dataset to load.
            config (Optional[str]): An optional configuration string for data loading.

        Returns:
            Any: The dataset loaded from the source.
        """
        dataset: Any = self._loader.load(dataset_name=dataset_name, config=config)
        try:
            logger.info("Dataset loaded with columns: %s", dataset)
            if hasattr(dataset, "keys") and callable(getattr(dataset, "keys", None)):
                for split_name in dataset.keys():
                    split_columns: Optional[Any] = getattr(
                        dataset[split_name], "column_names", None
                    )
                    logger.debug(
                        "Columns for split '%s': %s", split_name, split_columns
                    )
            else:
                logger.debug(
                    "Dataset columns: %s", getattr(dataset, "column_names", "Unknown")
                )
        except Exception as exc:
            logger.debug("Failed to log dataset columns: %s", exc, exc_info=True)
        return dataset

    def select_split(
        self, dataset: Any, config_obj: Optional[BaseDatasetConfig]
    ) -> Any:
        """Select a specific dataset split based on the provided configuration.

        Args:
            dataset (Any): A dataset that may contain multiple splits.
            config_obj (Optional[BaseDatasetConfig]): A configuration instance that may specify a split via its
                'split' attribute.

        Returns:
            Any: The selected dataset split if found; otherwise, the original dataset.
        """
        if config_obj is None or not hasattr(config_obj, "split"):
            return dataset
        split_name: Optional[str] = getattr(config_obj, "split", None)
        if split_name and split_name in dataset:
            return dataset[split_name]
        if split_name:
            logger.warning("Requested split '%s' not found.", split_name)
        return dataset

    def _validate_structure(self, dataset: Any) -> Any:
        """Validate the structural integrity of the provided dataset.

        Args:
            dataset (Any): The dataset to validate.

        Returns:
            Any: The validated dataset.
        """
        return self._validator.validate_structure(dataset=dataset)

    def _transform_data(self, data: Any) -> Any:
        """Apply a sequence of transformations to the input dataset.

        Each transformer in the configured list is applied in order.

        Args:
            data (Any): The input dataset to transform.

        Returns:
            Any: The dataset after applying all transformations.
        """
        transformed: Any = data
        for transformer in self._transformers:
            transformed = transformer.transform(data=transformed)
        return transformed

    def _validate_keys(self, data: Any, prepper: IDatasetPrepper) -> None:
        """Validate that a sample of dataset items contains the required keys.

        A random sample of items from the dataset is inspected.

        Args:
            data (Any): The dataset whose items will be validated.
            prepper (IDatasetPrepper): The prepper providing the set of required keys.
        """
        required_keys: List[str] = prepper.get_required_keys()
        sample_size: int = min(5, len(data))
        sample_indices: List[int] = random.sample(range(len(data)), sample_size)
        for idx in sample_indices:
            self._validator.validate_required_keys(
                item=data[idx], required_keys=required_keys
            )

    def _sample_data(self, data: Any, num_samples: Optional[int]) -> Any:
        """Downsample the dataset to a specified number of samples if requested.

        Args:
            data (Any): The dataset to be sampled.
            num_samples (Optional[int]): The desired number of samples; if None, returns the original dataset.

        Returns:
            Any: The downsampled dataset, or the original dataset if num_samples is None.
        """
        return self._sampler.sample(data=data, num_samples=num_samples)

    def _prep_data(
        self, dataset_info: DatasetInfo, sampled_data: Any, prepper: IDatasetPrepper
    ) -> List[DatasetEntry]:
        """Prepare and validate the final dataset entries from the supplied sampled data.

        Each item is validated and converted into one or more DatasetEntry objects. Malformed items are
        skipped with a warning logged.

        Args:
            dataset_info (DatasetInfo): Metadata describing the dataset.
            sampled_data (Any): The dataset after sampling.
            prepper (IDatasetPrepper): The prepper used for record validation and entry creation.

        Returns:
            List[DatasetEntry]: The list of final DatasetEntry objects.
        """
        entries: List[DatasetEntry] = []
        required_keys: List[str] = prepper.get_required_keys()
        for item in sampled_data:
            try:
                self._validator.validate_item(item=item, required_keys=required_keys)
                entries.extend(prepper.create_dataset_entries(item=item))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping malformed data from %s: %s. Item keys: %s; Required keys: %s",
                    dataset_info.name,
                    exc,
                    list(item),
                    required_keys,
                )
        return entries

    def load_and_prepare(
        self,
        dataset_info: DatasetInfo,
        prepper: IDatasetPrepper,
        config: Union[str, BaseDatasetConfig, None] = None,
        num_samples: Optional[int] = None,
    ) -> List[DatasetEntry]:
        """Execute the complete pipeline for processing and preparing a dataset.

        The pipeline includes:
          1. Converting the configuration for loader compatibility.
          2. Loading the dataset.
          3. Optionally selecting a specified dataset split.
          4. Validating the dataset structure.
          5. Applying data transformations.
          6. Validating required keys in the transformed data.
          7. Sampling the data if requested.
          8. Preparing the final dataset entries.

        Args:
            dataset_info (DatasetInfo): Metadata containing dataset details such as name and source.
            prepper (IDatasetPrepper): Instance responsible for final data preparation.
            config (Union[str, BaseDatasetConfig, None]): A configuration identifier for data loading.
            num_samples (Optional[int]): Desired number of samples; if None, processes the entire dataset.

        Returns:
            List[DatasetEntry]: The list of processed DatasetEntry objects ready for consumption.
        """
        logger.info(
            "[load_and_prepare] Starting process for dataset '%s' with dataset_name='%s', "
            "config='%s', num_samples='%s'.",
            dataset_info.name,
            dataset_info.source,
            config,
            num_samples,
        )

        logger.info(
            "[load_and_prepare] Converting configuration for loader compatibility."
        )
        resolved_config: Optional[str] = self._resolve_config_name(config=config)
        logger.info("[load_and_prepare] Resolved configuration: '%s'.", resolved_config)

        logger.info(
            "[load_and_prepare] Loading data from dataset_name='%s'...",
            dataset_info.source,
        )
        dataset: Any = self._load_data(
            dataset_name=dataset_info.source, config=resolved_config
        )
        logger.info("[load_and_prepare] Data loaded successfully.")

        if hasattr(dataset, "__len__"):
            logger.info(
                "[load_and_prepare] Dataset details: type=%s, size=%d",
                type(dataset),
                len(dataset),
            )
            if len(dataset) > 0:
                logger.info(
                    "[load_and_prepare] Sample record from dataset: %s", dataset
                )
        else:
            logger.info(
                "[load_and_prepare] Dataset type: %s, size: Unknown",
                type(dataset),
            )

        config_obj: Optional[BaseDatasetConfig] = (
            config if isinstance(config, BaseDatasetConfig) else None
        )
        logger.info(
            "[load_and_prepare] Configuration object is %s.",
            "a BaseDatasetConfig subclass" if config_obj else "None or a string",
        )

        logger.info("[load_and_prepare] Selecting requested split, if applicable.")
        dataset = self.select_split(dataset=dataset, config_obj=config_obj)
        logger.info("[load_and_prepare] Split selection completed.")

        logger.info("[load_and_prepare] Validating dataset structure.")
        validated_data: Any = self._validate_structure(dataset=dataset)
        logger.info("[load_and_prepare] Dataset structure validated successfully.")

        if hasattr(validated_data, "__len__"):
            logger.info(
                "[load_and_prepare] Validated dataset details: type=%s, size=%d",
                type(validated_data),
                len(validated_data),
            )
            if len(validated_data) > 0:
                logger.info(
                    "[load_and_prepare] Sample record from validated data: %s",
                    validated_data[0],
                )
        else:
            logger.info(
                "[load_and_prepare] Validated dataset type: %s, size: Unknown",
                type(validated_data),
            )

        logger.info("[load_and_prepare] Applying data transformations.")
        transformed_data: Any = self._transform_data(data=validated_data)
        logger.info("[load_and_prepare] Data transformations applied.")

        if hasattr(transformed_data, "__len__"):
            logger.info(
                "[load_and_prepare] Transformed dataset details: type=%s, size=%d",
                type(transformed_data),
                len(transformed_data),
            )
            if len(transformed_data) > 0:
                logger.info(
                    "[load_and_prepare] Sample record from transformed data: %s",
                    transformed_data[0],
                )
        else:
            logger.info(
                "[load_and_prepare] Transformed dataset type: %s, size: Unknown",
                type(transformed_data),
            )

        logger.info(
            "[load_and_prepare] Validating presence of required keys in the transformed data."
        )
        self._validate_keys(data=transformed_data, prepper=prepper)
        logger.info("[load_and_prepare] Required keys validated successfully.")

        logger.info("[load_and_prepare] Sampling data as specified.")
        sampled_data: Any = self._sample_data(
            data=transformed_data, num_samples=num_samples
        )
        logger.info(
            "[load_and_prepare] Sampling completed. Sampled data: %s", sampled_data
        )
        if hasattr(sampled_data, "__len__"):
            logger.info(
                "[load_and_prepare] Number of records after sampling: %d",
                len(sampled_data),
            )
            if len(sampled_data) > 0:
                logger.info(
                    "[load_and_prepare] Sample record from sampled data: %s",
                    sampled_data,
                )
        else:
            logger.info("[load_and_prepare] Number of records after sampling: Unknown")

        logger.info("[load_and_prepare] Preparing final dataset entries.")
        entries: List[DatasetEntry] = self._prep_data(
            dataset_info=dataset_info, sampled_data=sampled_data, prepper=prepper
        )
        logger.info(
            "[load_and_prepare] Preparation complete. Generated %d DatasetEntry objects.",
            len(entries),
        )
        return entries
