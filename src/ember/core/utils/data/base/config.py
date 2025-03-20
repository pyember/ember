from typing import Optional

from pydantic import BaseModel


class BaseDatasetConfig(BaseModel):
    """Generic dataset configuration model.

    This model encapsulates common dataset configuration parameters,
    including an optional identifier for a sub-dataset variation and the
    designated data split (e.g., 'train', 'test', or 'validation'). Subclasses
    should extend this model with additional, dataset-specific fields as needed.

    Attributes:
        config_name (Optional[str]): Optional identifier for a sub-dataset configuration.
            Defaults to None.
        split (Optional[str]): The data split to use for dataset processing.
            Defaults to 'train'.
    """

    config_name: Optional[str] = None
    split: Optional[str] = "train"
