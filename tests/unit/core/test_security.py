"""Security tests for credential and configuration handling.

Following security best practices:
- Never expose secrets in logs or errors
- Validate all external input
- Secure file permissions
- Prevent injection attacks
"""

import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from ember._internal.context import EmberContext
from ember.core.credentials import CredentialManager


class TestCredentialSecurity:
    """Credential storage security."""

    @unittest.skipIf(sys.platform.startswith("win"), "disabled on Windows")
    def test_file_permissions(self):
        """Credentials saved with restricted permissions in config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CredentialManager(Path(tmpdir))
            mgr.save_api_key("test", "secret-key")

            config_file = Path(tmpdir) / "config.yaml"
            assert config_file.exists()

            # Check permissions (owner read/write only)
            mode = config_file.stat().st_mode
            assert mode & stat.S_IRWXG == 0  # No group permissions
            assert mode & stat.S_IRWXO == 0  # No other permissions
            assert mode & stat.S_IRUSR != 0  # Owner can read
            assert mode & stat.S_IWUSR != 0  # Owner can write

    def test_no_credential_logging(self, caplog):
        """Credentials never appear in logs."""
        ctx = EmberContext(isolated=True)
        secret = "sk-super-secret-key-12345"

        # Various operations that might log
        ctx._credentials.save_api_key("openai", secret)
        ctx.get_credential("openai", "OPENAI_API_KEY")

        # Secret should never appear in logs
        assert secret not in caplog.text

    def test_atomic_writes(self):
        """Credentials use atomic writes to prevent corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CredentialManager(Path(tmpdir))

            # Write initial data
            mgr.save_api_key("test1", "test-key-1234")

            # Simulate concurrent write attempt
            with patch("tempfile.mkstemp") as mock_mkstemp:
                # Simulate write failure
                mock_mkstemp.side_effect = OSError("Disk full")

                # Should not corrupt existing data
                with pytest.raises(OSError):
                    mgr.save_api_key("test2", "test-key-5678")

                # Original data intact
                assert mgr.get("test1") == "test-key-1234"
