"""DSPy integration for Ember.

This module provides EmberLM, a custom language model backend for DSPy that
enables using any model in Ember's registry within DSPy programs.
"""

from ember.integrations.dspy.ember_lm import EmberLM

__all__ = ["EmberLM"]
