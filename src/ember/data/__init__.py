"""Datasets as typed examples.

`stream` reads a dataset from the Hugging Face Hub through the ``data`` extra
and yields `Example` records. Datasets ember knows by name are listed in
`DATASETS`; any other Hub dataset streams through a `Dataset` that names it and
maps its rows. Importing this package does not import ``datasets``.

Example:
    >>> from ember import data
    >>> sorted(data.DATASETS)
    ['gsm8k', 'mmlu']
"""

from ember.data.example import LETTERS, Dataset, Example, Row
from ember.data.named import DATASETS
from ember.data.stream import stream

__all__ = ["DATASETS", "LETTERS", "Dataset", "Example", "Row", "stream"]
