"""Tests for models CLI command."""

from unittest.mock import Mock, patch

from ember.cli.main import cmd_models
from ember.models.catalog import ModelInfo


class TestModelsCommand:
    """Test models command with clean mocking.

    Note: cmd_models uses ember.models.catalog functions directly:
    - get_providers() for listing providers
    - list_available_models() for listing models
    - get_model_info() for model descriptions
    """

    def test_list_providers(self, capsys):
        """List providers returns correct output."""
        with patch(
            "ember.models.catalog.get_providers",
            return_value=["openai", "anthropic", "google"],
        ):
            result = cmd_models(Mock(providers=True))

        output = capsys.readouterr().out
        assert result == 0
        assert "Available providers:" in output
        assert "- openai" in output
        assert "- anthropic" in output
        assert "- google" in output

    def test_list_models(self, capsys):
        """List models returns formatted output."""
        mock_info_gpt4 = ModelInfo(
            id="gpt-4",
            provider="openai",
            description="Advanced model",
            context_window=8192,
        )
        mock_info_claude = ModelInfo(
            id="claude-3",
            provider="anthropic",
            description="Anthropic model",
            context_window=100000,
        )

        def mock_get_info(model_id, include_dynamic=True):
            return {"gpt-4": mock_info_gpt4, "claude-3": mock_info_claude}[model_id]

        with (
            patch(
                "ember.models.catalog.list_available_models",
                return_value=["gpt-4", "claude-3"],
            ),
            patch("ember.models.catalog.get_model_info", side_effect=mock_get_info),
        ):
            result = cmd_models(Mock(providers=False, provider=None))

        output = capsys.readouterr().out
        assert result == 0
        assert "Available models:" in output
        assert "gpt-4" in output
        assert "Advanced model" in output
        assert "claude-3" in output
        assert "Anthropic model" in output

    def test_filter_by_provider(self, capsys):
        """Filter models by provider."""
        mock_info_gpt4 = ModelInfo(
            id="gpt-4",
            provider="openai",
            description="GPT-4",
            context_window=8192,
        )
        mock_info_gpt35 = ModelInfo(
            id="gpt-3.5",
            provider="openai",
            description="GPT-3.5",
            context_window=4096,
        )

        def mock_get_info(model_id, include_dynamic=True):
            return {"gpt-4": mock_info_gpt4, "gpt-3.5": mock_info_gpt35}[model_id]

        with (
            patch(
                "ember.models.catalog.list_available_models",
                return_value=["gpt-4", "gpt-3.5"],
            ) as mock_list,
            patch("ember.models.catalog.get_model_info", side_effect=mock_get_info),
        ):
            result = cmd_models(Mock(providers=False, provider="openai"))

            # Verify list_available_models was called with provider
            mock_list.assert_called_once_with(provider="openai", include_dynamic=False)

        output = capsys.readouterr().out
        assert result == 0
        assert "Available openai models:" in output
        assert "gpt-4" in output
        assert "gpt-3.5" in output


# Alternative approach using dependency injection
class TestModelsCommandDI:
    """Test with dependency injection pattern."""

    @patch("ember.cli.main.print")
    def test_providers_with_di(self, mock_print):
        """Test using print mocking instead of module mocking."""
        # This is a cleaner approach that doesn't require module manipulation
        mock_models = Mock()
        mock_models.providers = Mock(return_value=["provider1", "provider2"])

        # Patch at import location
        with patch.dict("sys.modules", {"ember.api": Mock(models=mock_models)}):
            result = cmd_models(Mock(providers=True))

        assert result == 0
        # Verify print was called correctly
        assert mock_print.call_count >= 3  # Header + 2 providers
