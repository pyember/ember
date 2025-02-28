"""Configure pytest environment for all tests."""

import pytest
from unittest.mock import patch
import sys
import os
from pathlib import Path

# Get absolute paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"

# Add both project root and src to path
for path in (str(PROJECT_ROOT), str(SRC_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# Add the src directory to the Python path
root_dir = Path(__file__).parent.parent.absolute()
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


@pytest.fixture(scope="session", autouse=True)
def global_setup_teardown():
    """
    Global fixture for session-level setup/teardown.
    - Can configure logging or environment variables here.
    """
    # TODO: placeholder to add global config as needed
    yield
    # TODO: placeholder to add global teardown as needed


@pytest.fixture
def mock_lm_generation(mocker):
    """
    Mocks LMModule model_instance.generate calls to return predictable responses.
    Ensures deterministic tests regardless of input prompt.
    """

    def mock_generate(prompt, temperature=1.0, max_tokens=None):
        return f"Mocked response: {prompt}, temp={temperature}"

    # Patch the DummyModel generate method.
    # Adjust the patch path if needed to reflect actual code location of get_model_registry usage.
    mocker.patch(
        "tests.get_model_registry().DummyModel.generate", side_effect=mock_generate
    )
