"""Golden file testing infrastructure."""

from .golden_file_manager import (
    GoldenFileManager,
    compare_golden,
    compare_golden_json,
    compare_golden_text,
    get_golden_manager,
)

__all__ = [
    "GoldenFileManager",
    "get_golden_manager",
    "compare_golden",
    "compare_golden_text",
    "compare_golden_json",
]
