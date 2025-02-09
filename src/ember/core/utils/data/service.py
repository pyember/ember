import logging
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
    """Service for orchestrating dataset operations including loading, validating,
    transforming, sampling, and preparing dataset entries.

    This service executes the following steps in sequence:
      1. Loads the dataset from a given source.
      2. Validates the overall structure of the dataset.
      3. Optionally selects a specific split.
      4. Applies sequential transformations.
      5. Validates the presence of required keys.
      6. Downsamples the dataset if needed.
      7. Prepares the final dataset entries.
    """

    def __init__(
        self,
        loader: IDatasetLoader,
        validator: IDatasetValidator,
        sampler: IDatasetSampler,
        transformers: Optional[Iterable[IDatasetTransformer]] = None,
    ) -> None:
        """Initializes the DatasetService instance.

        Args:
            loader (IDatasetLoader): An instance for loading datasets.
            validator (IDatasetValidator): An instance for validating dataset structures.
            sampler (IDatasetSampler): An instance for sampling dataset records.
            transformers (Optional[Iterable[IDatasetTransformer]]): Optional iterable of 
                transformers to be applied sequentially.
        """
        self._loader: IDatasetLoader = loader
        self._validator: IDatasetValidator = validator
        self._sampler: IDatasetSampler = sampler
        self._transformers: List[IDatasetTransformer] = list(transformers) if transformers else []

    def _resolve_loader_config(
        self, config: Union[str, BaseDatasetConfig, None]
    ) -> Optional[str]:
        """Resolves a configuration object into a string compatible with the loader.

        Args:
            config (Union[str, BaseDatasetConfig, None]): A configuration identifier provided
                as a string, a BaseDatasetConfig instance, or None.

        Returns:
            Optional[str]: A configuration string if resolvable; otherwise, None.
        """
        if isinstance(config, BaseDatasetConfig):
            return getattr(config, "config_name", None)
        if isinstance(config, str):
            return config
        return None

    def _load_data(self, source: str, config: Optional[str] = None) -> Any:
        """Loads data from the specified source using an optional configuration.

        Args:
            source (str): The source from which to load the dataset (e.g., file path, URL).
            config (Optional[str]): Optional configuration string for data loading.

        Returns:
            Any: The dataset loaded from the source.
        """
        dataset: Any = self._loader.load(source=source, config=config)
        try:
            logger.info("Dataset loaded with columns: %s", dataset)
            if hasattr(dataset, "keys") and callable(dataset.keys):
                for split_name in dataset.keys():
                    split_columns: Optional[Any] = getattr(dataset[split_name], "column_names", None)
                    logger.debug("Columns for split '%s': %s", split_name, split_columns)
            else:
                logger.debug("Dataset columns: %s", getattr(dataset, "column_names", "Unknown"))
        except Exception as exc:
            logger.debug("Failed to log dataset columns: %s", exc)
        return dataset

    def select_split(self, dataset: Any, config_obj: Optional[BaseDatasetConfig]) -> Any:
        """Selects a specific split from the dataset based on the provided configuration.

        If the configuration object contains a 'split' attribute and the dataset includes that
        split, then the specified split is returned. Otherwise, the original dataset is returned.

        Args:
            dataset (Any): The dataset, which may contain multiple splits.
            config_obj (Optional[BaseDatasetConfig]): Configuration instance that may specify a split.

        Returns:
            Any: The selected dataset split if available; otherwise, the original dataset.
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
        """Validates the structural integrity of the dataset.

        Args:
            dataset (Any): The dataset to be validated.

        Returns:
            Any: The dataset after successful structural validation.
        """
        return self._validator.validate_structure(dataset=dataset)

    def _transform_data(self, data: Any) -> Any:
        """Applies a series of transformations to the dataset.

        Each transformer in the configured list is sequentially applied to the data.

        Args:
            data (Any): The input dataset to be transformed.

        Returns:
            Any: The dataset after all transformations have been applied.
        """
        transformed: Any = data
        for transformer in self._transformers:
            transformed = transformer.transform(data=transformed)
        return transformed

    def _validate_keys(self, data: Any, prepper: IDatasetPrepper) -> None:
        """Validates that the first record in the dataset contains the required keys.

        This method extracts the first item from the dataset and checks for required keys using
        the provided prepper.

        Args:
            data (Any): The dataset from which the first record is checked.
            prepper (IDatasetPrepper): The instance supplying the required keys list.

        Raises:
            KeyError, ValueError, or TypeError: Propagates any exceptions raised during validation.
        """
        first_item: Any = data[0]
        required_keys: List[str] = prepper.get_required_keys()
        self._validator.validate_required_keys(item=first_item, required_keys=required_keys)

    def _sample_data(self, data: Any, num_samples: Optional[int]) -> Any:
        """Downsamples the dataset to a specified number of samples if provided.

        Args:
            data (Any): The dataset to sample.
            num_samples (Optional[int]): The number of samples desired; if None, the data is unchanged.

        Returns:
            Any: The subset of the dataset after sampling.
        """
        return self._sampler.sample(data=data, num_samples=num_samples)

    def _prep_data(
        self, dataset_info: DatasetInfo, sampled_data: Any, prepper: IDatasetPrepper
    ) -> List[DatasetEntry]:
        """Prepares the final dataset entries from the sampled data.

        Each record is validated and transformed into one or more DatasetEntry objects.

        Args:
            dataset_info (DatasetInfo): Metadata describing the dataset.
            sampled_data (Any): The dataset after sampling.
            prepper (IDatasetPrepper): The instance used for final record validation and entry creation.

        Returns:
            List[DatasetEntry]: A list of the final DatasetEntry objects.
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
        """Executes the complete pipeline for processing and preparing a dataset.

        The pipeline encompasses configuration resolution, data loading, optional split selection,
        structure validation, data transformation, required key validation, optional sampling,
        and final dataset entry preparation.

        Args:
            dataset_info (DatasetInfo): Metadata object containing dataset details such as name and source.
            prepper (IDatasetPrepper): An instance responsible for the final data preparation steps.
            config (Union[str, BaseDatasetConfig, None]): A configuration identifier for data loading.
            num_samples (Optional[int]): The desired number of samples; if None, the entire dataset is used.

        Returns:
            List[DatasetEntry]: A list of processed DatasetEntry objects ready for further consumption.
        """
        logger.info(
            "[load_and_prepare] Starting process for dataset '%s' with source='%s', "
            "config='%s', num_samples='%s'.",
            dataset_info.name,
            dataset_info.source,
            config,
            num_samples,
        )

        logger.info("[load_and_prepare] Converting configuration for loader compatibility.")
        resolved_config: Optional[str] = self._resolve_loader_config(config=config)
        logger.info("[load_and_prepare] Resolved configuration: '%s'.", resolved_config)

        logger.info(
            "[load_and_prepare] Loading data from source='%s' with resolved configuration.",
            dataset_info.source,
        )
        dataset: Any = self._load_data(source=dataset_info.source, config=resolved_config)
        logger.info("[load_and_prepare] Data loaded successfully.")

        if hasattr(dataset, "__len__"):
            logger.info(
                "[load_and_prepare] Dataset details: type=%s, size=%d",
                type(dataset),
                len(dataset),
            )
            if len(dataset) > 0:
                logger.info("[load_and_prepare] Sample record from dataset: %s", dataset)
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
        sampled_data: Any = self._sample_data(data=transformed_data, num_samples=num_samples)
        logger.info("[load_and_prepare] Sampling completed. Sampled data: %s", sampled_data)
        if hasattr(sampled_data, "__len__"):
            logger.info(
                "[load_and_prepare] Number of records after sampling: %d", len(sampled_data)
            )
            if len(sampled_data) > 0:
                logger.info(
                    "[load_and_prepare] Sample record from sampled data: %s", sampled_data
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
