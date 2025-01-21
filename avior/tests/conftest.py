import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def global_setup_teardown():
    """
    Global fixture for session-level setup/teardown.
    - Can configure logging or environment variables here.
    """
    # TODO: Add global config if needed
    yield
    # TODO: Global teardown if needed


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
