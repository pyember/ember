import logging
from typing import Optional, Iterable, List, Dict, Any, Union
from src.avior.registry.dataset.base.loaders import IDatasetLoader
from src.avior.registry.dataset.base.validators import IDatasetValidator
from src.avior.registry.dataset.base.samplers import IDatasetSampler
from src.avior.registry.dataset.base.transformers import IDatasetTransformer
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetInfo, DatasetEntry
from src.avior.registry.dataset.base.config import BaseDatasetConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DatasetService:
    def __init__(
        self,
        loader: IDatasetLoader,
        validator: IDatasetValidator,
        sampler: IDatasetSampler,
        transformers: Optional[Iterable[IDatasetTransformer]] = None,
    ) -> None:
        self._loader = loader
        self._validator = validator
        self._sampler = sampler
        self._transformers = list(transformers) if transformers else []

    def _resolve_loader_config(
        self, config: Union[str, BaseDatasetConfig, None]
    ) -> Optional[str]:
        """
        Convert whatever 'config' object we receive into a string
        suitable for the loader. This is intentionally simple:
          - If it's a string, return it
          - If it's a BaseDatasetConfig with e.g. 'config_name', return that
          - Otherwise, return None
        """
        if isinstance(config, BaseDatasetConfig):
            return getattr(config, "config_name", None)
        if isinstance(config, str):
            return config
        return None

    def _load_data(self, source: str, config: Optional[str] = None) -> Any:
        """
        Step 1: Load the dataset from a given source and config.
        """
        dataset = self._loader.load(source, config)
        try:
            logger.info(f"Dataset columns: {dataset}")
            if hasattr(dataset, "keys") and callable(dataset.keys):
                for split_name in dataset.keys():
                    split_columns = dataset[split_name].column_names
                    logger.debug(f"Split '{split_name}' columns: {split_columns}")
            else:
                logger.debug(f"Dataset columns: {dataset.column_names}")
        except Exception as exc:
            logger.debug(f"Could not log dataset columns: {exc}")
        return dataset

    def select_split(self, dataset: Any, config_obj: Optional[BaseDatasetConfig]) -> Any:
        """
        If config_obj has a 'split' attribute (like MMLUConfig does),
        then try dataset[split]. Otherwise, return dataset unchanged.
        """
        if not config_obj or not hasattr(config_obj, "split"):
            return dataset
        split_name = getattr(config_obj, "split", None)
        if split_name and split_name in dataset:
            return dataset[split_name]
        if split_name:
            logger.warning(f"Requested split '{split_name}' not found.")
        return dataset

    def _validate_structure(self, dataset: Any) -> Any:
        """
        Step 2: Validate dataset structure.
        """
        return self._validator.validate_structure(dataset)

    def _transform_data(self, data: Any) -> Any:
        """
        Step 3: Apply any configured transformers in sequence.
        """
        transformed = data
        for transformer in self._transformers:
            transformed = transformer.transform(transformed)
        return transformed

    def _validate_keys(self, data: Any, prepper: IDatasetPrepper) -> None:
        """
        Step 4: Confirm required keys exist in a sample item.
        """
        first_item = data[0]
        self._validator.validate_required_keys(first_item, prepper.get_required_keys())

    def _sample_data(self, data: Any, num_samples: Optional[int]) -> Any:
        """
        Step 5: (Optional) downsample the data.
        """
        return self._sampler.sample(data, num_samples)

    def _prep_data(
        self,
        dataset_info: DatasetInfo,
        sampled_data: Any,
        prepper: IDatasetPrepper
    ) -> List[DatasetEntry]:
        """
        Step 6: Final prep, validating each item and creating DatasetEntry objects.
        """
        entries: List[DatasetEntry] = []
        for item in sampled_data:
            try:
                self._validator.validate_item(item, prepper.get_required_keys())
                entries.extend(prepper.create_dataset_entries(item))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Skipping malformed data from {dataset_info.name}: {e}. "
                    f"Item keys found: {list(item)}, "
                    f"Required keys: {prepper.get_required_keys()}"
                )
        return entries

    def load_and_prepare(
        self,
        dataset_info: DatasetInfo,
        prepper: IDatasetPrepper,
        config: Union[str, BaseDatasetConfig, None] = None,
        num_samples: Optional[int] = None,
    ) -> List[DatasetEntry]:
        logging.info(
            f"[load_and_prepare] Starting process for dataset '{dataset_info.name}' "
            f"with source='{dataset_info.source}', config='{config}', num_samples='{num_samples}'."
        )

        logging.info("[load_and_prepare] Converting config -> a string (or None) that _load_data expects.")
        resolved_config = self._resolve_loader_config(config)
        logging.info(f"[load_and_prepare] Resolved config is: '{resolved_config}'.")

        logging.info(
            f"[load_and_prepare] Loading data from source='{dataset_info.source}' "
            f"using resolved_config='{resolved_config}'."
        )
        dataset = self._load_data(dataset_info.source, resolved_config)
        logging.info("[load_and_prepare] Data loaded successfully.")
        if hasattr(dataset, "__len__"):
            logging.info(
                f"[load_and_prepare] Loaded dataset details: type={type(dataset)}, size={len(dataset)}"
            )
            if len(dataset) > 0:
                logging.info(f"[load_and_prepare] Example record from loaded dataset: {dataset}")
        else:
            logging.info(f"[load_and_prepare] Loaded dataset is of type={type(dataset)}, size=Unknown")

        config_obj = config if isinstance(config, BaseDatasetConfig) else None
        logging.info(
            "[load_and_prepare] Determined that 'config_obj' is "
            f"{'a BaseDatasetConfig subclass' if config_obj else 'None or string'}."
        )

        logging.info("[load_and_prepare] Selecting appropriate split, if requested.")
        dataset = self.select_split(dataset, config_obj)
        logging.info("[load_and_prepare] Split selection completed.")

        logging.info("[load_and_prepare] Validating dataset structure.")
        validated_data = self._validate_structure(dataset)
        logging.info("[load_and_prepare] Dataset structure validated successfully.")
        if hasattr(validated_data, "__len__"):
            logging.info(
                f"[load_and_prepare] Validated data details: type={type(validated_data)}, size={len(validated_data)}"
            )
            if len(validated_data) > 0:
                logging.info(f"[load_and_prepare] Example record from validated data: {validated_data[0]}")
        else:
            logging.info(f"[load_and_prepare] Validated data is of type={type(validated_data)}, size=Unknown")

        logging.info("[load_and_prepare] Applying transformations to the validated data.")
        transformed_data = self._transform_data(validated_data)
        logging.info("[load_and_prepare] Data transformations applied.")
        if hasattr(transformed_data, "__len__"):
            logging.info(
                f"[load_and_prepare] Transformed data details: type={type(transformed_data)}, size={len(transformed_data)}"
            )
            if len(transformed_data) > 0:
                logging.info(f"[load_and_prepare] Example record from transformed data: {transformed_data[0]}")
        else:
            logging.info(f"[load_and_prepare] Transformed data is of type={type(transformed_data)}, size=Unknown")

        logging.info("[load_and_prepare] Validating required keys in the transformed data.")
        self._validate_keys(transformed_data, prepper)
        logging.info("[load_and_prepare] Required keys validation completed.")

        logging.info("[load_and_prepare] Sampling the data if num_samples is provided.")
        logging.info(f"[load_and_prepare] Transformed data: {transformed_data}")
        sampled_data = self._sample_data(transformed_data, num_samples)
        logging.info(f"[load_and_prepare] Sampled data: {sampled_data}")
        logging.info(
            f"[load_and_prepare] Data sampling complete."
            f" Number of records after sampling: "
            f"{len(sampled_data) if hasattr(sampled_data, '__len__') else 'Unknown'}."
        )
        if hasattr(sampled_data, "__len__") and len(sampled_data) > 0:
            logging.info(f"[load_and_prepare] Example record from sampled data: {sampled_data}")

        logging.info("[load_and_prepare] Preparing final DatasetEntry objects.")
        entries = self._prep_data(dataset_info, sampled_data, prepper)
        logging.info(f"[load_and_prepare] Final preparation complete. Generated {len(entries)} DatasetEntry objects.")

        return entries

