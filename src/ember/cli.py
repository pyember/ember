#!/usr/bin/env python
"""
Ember CLI - Command Line Interface for Ember
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

import ember


def version_cmd(args: argparse.Namespace) -> None:
    """Display version information."""
    print(f"Ember version: {ember.__version__}")


def providers_cmd(args: argparse.Namespace) -> None:
    """List available model providers."""
    service = ember.init(usage_tracking=False)
    providers = service.registry.list_providers()
    
    if not providers:
        print("No providers registered.")
        return
    
    print("Available providers:")
    for provider in sorted(providers):
        print(f"  • {provider}")


def models_cmd(args: argparse.Namespace) -> None:
    """List available models, optionally filtering by provider."""
    service = ember.init(usage_tracking=False)
    models = service.registry.list_models()
    
    if args.provider:
        models = [m for m in models if m.startswith(f"{args.provider}:")]
    
    if not models:
        if args.provider:
            print(f"No models found for provider '{args.provider}'.")
        else:
            print("No models registered.")
        return
    
    print("Available models:")
    for model in sorted(models):
        print(f"  • {model}")


def quickstart_cmd(args: argparse.Namespace) -> None:
    """Create a quickstart project."""
    dir_name = args.project_name
    
    if os.path.exists(dir_name):
        print(f"Error: Directory '{dir_name}' already exists.")
        return
    
    # Create project directory
    os.makedirs(dir_name)
    
    # Create simple example file
    example_file = os.path.join(dir_name, "ember_example.py")
    with open(example_file, "w") as f:
        f.write("""#!/usr/bin/env python
'''
Ember Quickstart Example
'''

import ember
from typing import ClassVar

from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core import non


# Define structured input/output types
class QueryInput(EmberModel):
    query: str
    
class QueryOutput(EmberModel):
    answer: str
    

# Define the specification
class SimpleQASpecification(Specification):
    input_model = QueryInput
    output_model = QueryOutput
    prompt_template = "Please answer this question: {query}"


# Create a simple operator
class SimpleQA(Operator[QueryInput, QueryOutput]):
    # Class-level specification declaration
    specification: ClassVar[Specification] = SimpleQASpecification()
    
    # Class-level field declarations
    lm: non.UniformEnsemble
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        # Initialize fields
        self.lm = non.UniformEnsemble(
            num_units=1,
            model_name=model_name
        )
    
    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        # Process the query through the LLM
        response = self.lm(inputs={"query": inputs.query})
        
        # Return structured output
        return QueryOutput(answer=response["responses"][0])


def main():
    '''Main entry point'''
    # Initialize Ember with one line
    service = ember.init()
    
    # Create and use our QA system
    qa = SimpleQA()
    result = qa(inputs={"query": "What is the capital of France?"})
    
    print(f"Answer: {result.answer}")
    
    # Direct model invocation is also possible
    direct_response = service("openai:gpt-4o-mini", "What is 2+2?")
    print(f"Direct response: {direct_response.data}")


if __name__ == "__main__":
    main()
""")
    
    # Create README.md
    readme_file = os.path.join(dir_name, "README.md")
    with open(readme_file, "w") as f:
        f.write(f"""# {dir_name}

A project created with Ember CLI.

## Setup

1. Install dependencies:
```bash
pip install "ember-ai[openai]"
```

2. Set up your API keys:
```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"

# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
```

## Run the example

```bash
python ember_example.py
```
""")
    
    # Create .env.example file
    env_file = os.path.join(dir_name, ".env.example")
    with open(env_file, "w") as f:
        f.write("""# API Keys - Replace with your actual keys and rename to .env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
""")
    
    print(f"Created Ember project in '{dir_name}'")
    print(f"To get started:")
    print(f"  cd {dir_name}")
    print(f"  pip install 'ember-ai[openai]'")
    print(f"  # Set your API keys")
    print(f"  python ember_example.py")


def invoke_cmd(args: argparse.Namespace) -> None:
    """Invoke an LLM model with a prompt."""
    if not args.model:
        print("Error: Please specify a model ID with --model")
        return
    
    if not args.prompt:
        print("Error: Please specify a prompt with --prompt")
        return
    
    service = ember.init(usage_tracking=True)
    
    try:
        response = service(args.model, args.prompt)
        print(response.data)
        
        if args.show_usage:
            usage = service.usage_service.get_last_usage()
            if usage:
                print("\nUsage:")
                print(f"  Tokens: {usage.tokens}")
                print(f"  Cost: ${usage.cost:.6f}")
    except Exception as e:
        print(f"Error: {str(e)}")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Ember CLI - Command line tools for Ember",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add version info
    parser.add_argument("--version", action="store_true", help="Show version information")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=version_cmd)
    
    # Providers command
    providers_parser = subparsers.add_parser("providers", help="List available providers")
    providers_parser.set_defaults(func=providers_cmd)
    
    # Models command
    models_parser = subparsers.add_parser("models", help="List available models")
    models_parser.add_argument("--provider", help="Filter models by provider")
    models_parser.set_defaults(func=models_cmd)
    
    # Quickstart command
    quickstart_parser = subparsers.add_parser("quickstart", help="Create a new Ember project")
    quickstart_parser.add_argument("project_name", help="Name of the project directory")
    quickstart_parser.set_defaults(func=quickstart_cmd)
    
    # Invoke command
    invoke_parser = subparsers.add_parser("invoke", help="Invoke a model with a prompt")
    invoke_parser.add_argument("--model", "-m", help="Model ID to use")
    invoke_parser.add_argument("--prompt", "-p", help="Prompt to send to the model")
    invoke_parser.add_argument("--show-usage", "-u", action="store_true", help="Show token usage")
    invoke_parser.set_defaults(func=invoke_cmd)
    
    # Parse args
    args = parser.parse_args()
    
    # Handle version flag at top level
    if args.version:
        version_cmd(args)
        return
    
    # If no command is provided, show help
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    # Execute the corresponding function
    args.func(args)


if __name__ == "__main__":
    main()