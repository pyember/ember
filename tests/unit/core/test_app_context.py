"""
Test the EmberContext and EmberAppContext functionality.
"""

import unittest
from unittest.mock import patch, MagicMock

from ember.core.app_context import (
    EmberAppContext,
)


class TestEmberAppContext(unittest.TestCase):
    def test_app_context_initialization(self):
        """Test that EmberAppContext initializes correctly with required dependencies."""
        # Arrange
        config_manager = MagicMock()
        model_registry = MagicMock()
        usage_service = MagicMock()
        logger = MagicMock()

        # Act
        app_context = EmberAppContext(
            config_manager=config_manager,
            model_registry=model_registry,
            usage_service=usage_service,
            logger=logger,
        )

        # Assert
        self.assertEqual(app_context.config_manager, config_manager)
        self.assertEqual(app_context.model_registry, model_registry)
        self.assertEqual(app_context.usage_service, usage_service)
        self.assertEqual(app_context.logger, logger)
