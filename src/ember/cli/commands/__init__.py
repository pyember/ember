"""Ember CLI command implementations.

This package contains individual command implementations for the Ember CLI,
following a modular design where each command is self-contained with its own
argument parsing, validation, and execution logic.

Architecture Principles:
    Each command module implements a standard interface:
    1. **add_parser(subparsers)**: Registers command with argparse
    2. **run(args, context)**: Executes command with parsed args and context
    3. **Isolated Dependencies**: Commands import only what they need
    4. **Fail Fast**: Validate inputs early with clear error messages

Command Design Patterns:
    - **Consistent Naming**: Verb-noun pattern (e.g., 'configure get', 'models list')
    - **Unix Philosophy**: Each command does one thing well
    - **Composability**: Commands can be piped and scripted
    - **Testability**: Pure functions where possible, clear side effects

Available Commands:
    - configure: Manage Ember configuration (get/set/list/remove)
    - models: Discover and test model providers
    - setup: Interactive configuration wizard
    - registry: Manage model registry entries
    - context: Inspect and manipulate context hierarchy

Performance Strategy:
    Commands are lazy-loaded to minimize startup overhead. Each command
    module is imported only when that specific command is invoked, keeping
    'ember --help' fast while supporting rich functionality.

Error Handling Philosophy:
    - User errors (invalid input): Clear messages, exit code 2
    - System errors (network, permissions): Diagnostic info, exit code 1
    - Interrupts (Ctrl-C): Clean shutdown, exit code 130
    - Success: Silent or minimal output, exit code 0
"""
