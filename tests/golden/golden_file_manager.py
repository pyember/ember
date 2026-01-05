"""Golden file management helper."""

import difflib
import json
import os
from pathlib import Path
from typing import Any, Optional, Union


class GoldenFileManager:
    """Manages golden files for test comparisons.

    Golden files are reference outputs that tests compare against.
    This manager handles reading, writing, and comparing golden files
    with clear workflows for updating when needed.
    """

    def __init__(self, base_path: Union[str, Path]):
        """Initialize with base directory for golden files.

        Args:
            base_path: Directory where golden files are stored
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Track whether we're in update mode
        self._update_mode = os.environ.get("UPDATE_GOLDEN_FILES", "").lower() == "true"

    @property
    def update_mode(self) -> bool:
        """Check if golden files should be updated.

        Set UPDATE_GOLDEN_FILES=true environment variable to update.
        """
        return self._update_mode

    def get_path(self, test_name: str, suffix: str = ".json") -> Path:
        """Get path for a golden file.

        Args:
            test_name: Name of the test (used as filename)
            suffix: File extension (default: .json)

        Returns:
            Path to the golden file
        """
        # Sanitize test name for filesystem
        safe_name = test_name.replace("/", "_").replace("\\", "_")
        return self.base_path / f"{safe_name}{suffix}"

    def load(self, test_name: str, suffix: str = ".json") -> Any:
        """Load golden file contents.

        Args:
            test_name: Name of the test
            suffix: File extension

        Returns:
            Parsed contents of golden file

        Raises:
            FileNotFoundError: If golden file doesn't exist
        """
        path = self.get_path(test_name, suffix)

        if not path.exists():
            raise FileNotFoundError(
                f"Golden file not found: {path}\nRun with UPDATE_GOLDEN_FILES=true to create it."
            )

        if suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        else:
            with open(path, "r") as f:
                return f.read()

    def save(self, test_name: str, data: Any, suffix: str = ".json") -> None:
        """Save data as golden file.

        Args:
            test_name: Name of the test
            data: Data to save
            suffix: File extension
        """
        path = self.get_path(test_name, suffix)

        if suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
        else:
            with open(path, "w") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))

    def compare(self, test_name: str, actual: Any, suffix: str = ".json") -> None:
        """Compare actual output with golden file.

        Args:
            test_name: Name of the test
            actual: Actual output to compare
            suffix: File extension

        Raises:
            AssertionError: If outputs don't match (with diff)
        """
        if self.update_mode:
            # Update golden file
            self.save(test_name, actual, suffix)
            print(f"Updated golden file: {self.get_path(test_name, suffix)}")
            return

        try:
            expected = self.load(test_name, suffix)
        except FileNotFoundError:
            # First run - create golden file
            self.save(test_name, actual, suffix)
            print(f"Created golden file: {self.get_path(test_name, suffix)}")
            return

        # Compare
        if suffix == ".json":
            # Normalize JSON for comparison
            actual_str = json.dumps(actual, indent=2, sort_keys=True)
            expected_str = json.dumps(expected, indent=2, sort_keys=True)
        else:
            actual_str = str(actual)
            expected_str = str(expected)

        if actual_str != expected_str:
            # Generate diff
            diff = difflib.unified_diff(
                expected_str.splitlines(keepends=True),
                actual_str.splitlines(keepends=True),
                fromfile=f"{test_name}.expected",
                tofile=f"{test_name}.actual",
                lineterm="",
            )

            diff_text = "".join(diff)
            raise AssertionError(
                f"Output doesn't match golden file.\n\n"
                f"Diff:\n{diff_text}\n\n"
                f"Run with UPDATE_GOLDEN_FILES=true to update golden file."
            )

    def compare_text(self, test_name: str, actual: str) -> None:
        """Compare text output with golden file.

        Convenience method for text comparisons.
        """
        self.compare(test_name, actual, suffix=".txt")

    def compare_json(self, test_name: str, actual: Any) -> None:
        """Compare JSON-serializable output with golden file.

        Convenience method for JSON comparisons.
        """
        self.compare(test_name, actual, suffix=".json")


# Global instance for convenience
_default_manager: Optional[GoldenFileManager] = None


def get_golden_manager(base_path: Optional[Union[str, Path]] = None) -> GoldenFileManager:
    """Get or create the default golden file manager.

    Args:
        base_path: Base directory for golden files.
                   Defaults to tests/golden/files/

    Returns:
        GoldenFileManager instance
    """
    global _default_manager

    if _default_manager is None or base_path is not None:
        if base_path is None:
            # Default to tests/golden/files/
            base_path = Path(__file__).parent / "files"
        _default_manager = GoldenFileManager(base_path)

    return _default_manager


# Convenience functions using default manager
def compare_golden(test_name: str, actual: Any, suffix: str = ".json") -> None:
    """Compare output with golden file using default manager."""
    manager = get_golden_manager()
    manager.compare(test_name, actual, suffix)


def compare_golden_text(test_name: str, actual: str) -> None:
    """Compare text output with golden file."""
    manager = get_golden_manager()
    manager.compare_text(test_name, actual)


def compare_golden_json(test_name: str, actual: Any) -> None:
    """Compare JSON output with golden file."""
    manager = get_golden_manager()
    manager.compare_json(test_name, actual)
