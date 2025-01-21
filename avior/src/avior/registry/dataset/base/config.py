from pydantic import BaseModel
from typing import Optional

class BaseDatasetConfig(BaseModel):
    """
    A generic dataset config class. Subclasses can add fields as needed.
    """
    # We can define shared fields like "split" or "version" here if desired.
    pass