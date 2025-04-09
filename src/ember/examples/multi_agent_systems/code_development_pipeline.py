"""Multi-Agent Code Development Pipeline.

This module demonstrates a multi-agent system for developing software components.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeDevelopmentPipeline:
    """A multi-agent system for collaborative code development.
    
    This pipeline uses specialized agents to handle different aspects of code development:
    - Requirements Analyst
    - System Architect
    - Code Generator
    - Style Guide Expert
    - Code Formatter
    - Code Reviewer
    - Test Writer
    - Documentation Writer
    """

    def __init__(self):
        """Initialize the specialized agents with different model configurations."""
        # All agents using OpenAI models with different specializations
        self.requirements_agent = models.model("openai:gpt-4-turbo", temperature=0.2)
        self.architect_agent = models.model("openai:gpt-4-turbo", temperature=0.3)
        self.code_generator = models.model("openai:gpt-4-turbo", temperature=0.2)
        self.style_guide_agent = models.model("openai:gpt-4-turbo", temperature=0.1)
        self.code_formatter = models.model("openai:gpt-4-turbo", temperature=0.1)
        self.code_reviewer = models.model("openai:gpt-4-turbo", temperature=0.1)
        self.test_writer = models.model("openai:gpt-4-turbo", temperature=0.2)
        self.documentation_writer = models.model("openai:gpt-4-turbo", temperature=0.3)

    def analyze_requirements(self, project_description: str) -> Dict[str, Any]:
        """Analyze project requirements from a project description.
        
        Args:
            project_description: A description of the project
            
        Returns:
            Dictionary of analyzed requirements
        """
        logger.info("Analyzing requirements...")
        prompt = f"""Analyze the following project description and extract key requirements:
        
Project Description:
{project_description}

Your task:
1. Identify all functional requirements
2. Identify all non-functional requirements
3. Identify key user stories
4. List potential technical constraints
5. Highlight any ambiguities that need clarification

Format your response as JSON with these categories.
"""
        result = self.requirements_agent(prompt)
        return {"requirements_analysis": result}

    def design_architecture(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design a system architecture based on the analyzed requirements.
        
        Args:
            requirements: Dict containing analyzed requirements
            
        Returns:
            Dictionary with architectural design
        """
        logger.info("Designing architecture...")
        prompt = f"""Based on the following requirements analysis, design a high-level architecture:

Requirements Analysis:
{requirements["requirements_analysis"]}

Your task:
1. Identify key components/modules
2. Define interfaces between components
3. Specify data models/structures
4. Recommend technologies/frameworks
5. Create a diagram description (ASCII or text-based)

Format your response as JSON with these categories.
"""
        result = self.architect_agent(prompt)
        return {"architecture_design": result}

    def generate_code(self, requirements: Dict[str, Any], architecture: Dict[str, Any], component_name: str) -> Dict[str, Any]:
        """Generate code for a specific component based on requirements and architecture.
        
        Args:
            requirements: Dict containing analyzed requirements
            architecture: Dict containing architectural design
            component_name: Name of the component to generate
            
        Returns:
            Dictionary with the generated code
        """
        logger.info(f"Generating code for {component_name}...")
        prompt = f"""Generate code for the {component_name} component based on:

Requirements Analysis:
{requirements["requirements_analysis"]}

Architecture Design:
{architecture["architecture_design"]}

Your task:
1. Write production-quality code for the {component_name} component
2. Include necessary imports
3. Add brief inline comments for complex logic
4. Implement all required methods/functions
5. Handle basic error cases
6. Focus on readability and maintainability

Format your response as valid code with no additional text.
"""
        result = self.code_generator(prompt)
        return {"initial_code": result}

    def apply_code_style(self, code: Dict[str, Any], language: str, style_guide: str) -> Dict[str, Any]:
        """Apply code style guidelines to the generated code.
        
        Args:
            code: Dict containing initial code
            language: Programming language of the code
            style_guide: Style guide to follow (e.g., PEP 8 for Python)
            
        Returns:
            Dictionary with styled code
        """
        logger.info(f"Applying {style_guide} style to code...")
        prompt = f"""Apply the {style_guide} style guide to the following {language} code:

```{language}
{code["initial_code"]}
```

Your task:
1. Follow {style_guide} conventions
2. Fix any style issues (naming, spacing, indentation, etc.)
3. Improve code organization if needed
4. Add or improve docstrings/comments as needed
5. Don't change the functionality

Return only the updated code with no additional text.
"""
        result = self.style_guide_agent(prompt)
        return {"styled_code": result}

    def format_code(self, styled_code: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Format the code for consistency and readability.
        
        Args:
            styled_code: Dict containing styled code
            language: Programming language of the code
            
        Returns:
            Dictionary with formatted code
        """
        logger.info("Formatting code for consistency...")
        prompt = f"""Format the following {language} code for optimal readability and consistency:

```{language}
{styled_code["styled_code"]}
```

Your task:
1. Apply consistent indentation
2. Apply consistent line spacing between methods/functions
3. Apply consistent spacing around operators
4. Apply consistent braces/block style
5. Ensure line length follows best practices
6. Keep the existing functionality intact

Return only the formatted code with no additional text.
"""
        result = self.code_formatter(prompt)
        return {"formatted_code": result}

    def review_code(self, formatted_code: Dict[str, Any], requirements: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Review the code for issues and suggest improvements.
        
        Args:
            formatted_code: Dict containing formatted code
            requirements: Dict containing analyzed requirements
            language: Programming language of the code
            
        Returns:
            Dictionary with review comments and improved code
        """
        logger.info("Reviewing code...")
        prompt = f"""Review the following {language} code against the requirements and best practices:

```{language}
{formatted_code["formatted_code"]}
```

Requirements:
{requirements["requirements_analysis"]}

Your task:
1. Identify any bugs or logical errors
2. Check for security vulnerabilities
3. Evaluate performance issues
4. Verify the code meets requirements
5. Suggest specific improvements
6. Provide an improved version of the code

Format your response as JSON with "review_comments" and "improved_code" fields.
"""
        result = self.code_reviewer(prompt)
        return {"code_review": result}

    def write_tests(self, formatted_code: Dict[str, Any], requirements: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Write tests for the code.
        
        Args:
            formatted_code: Dict containing formatted code
            requirements: Dict containing analyzed requirements
            language: Programming language of the code
            
        Returns:
            Dictionary with test code
        """
        logger.info("Writing tests...")
        prompt = f"""Write comprehensive tests for the following {language} code:

```{language}
{formatted_code["formatted_code"]}
```

Requirements:
{requirements["requirements_analysis"]}

Your task:
1. Write unit tests covering all functions/methods
2. Include both positive and negative test cases
3. Test edge cases
4. Ensure test code follows {language} best practices
5. Include brief comments explaining the purpose of each test
6. Use appropriate testing framework for {language}

Return only the test code with no additional text.
"""
        result = self.test_writer(prompt)
        return {"tests": result}

    def write_documentation(self, formatted_code: Dict[str, Any], architecture: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Write documentation for the code.
        
        Args:
            formatted_code: Dict containing formatted code
            architecture: Dict containing architectural design
            language: Programming language of the code
            
        Returns:
            Dictionary with documentation
        """
        logger.info("Writing documentation...")
        prompt = f"""Write comprehensive documentation for the following {language} code:

```{language}
{formatted_code["formatted_code"]}
```

Architecture Context:
{architecture["architecture_design"]}

Your task:
1. Create a README with overview and usage examples
2. Document the component's purpose and integration with other components
3. Document all public APIs, functions, and classes
4. Include installation and configuration instructions
5. Document any dependencies
6. Provide troubleshooting information

Return the documentation in Markdown format.
"""
        result = self.documentation_writer(prompt)
        return {"documentation": result}

    def develop_component(self, project_description: str, component_name: str, language: str, style_guide: str) -> Dict[str, Any]:
        """Execute the full code development pipeline.
        
        Args:
            project_description: A description of the project
            component_name: Name of the component to develop
            language: Programming language to use
            style_guide: Style guide to follow
            
        Returns:
            Dictionary with all artifacts from the development process
        """
        # Analyze requirements
        requirements = self.analyze_requirements(project_description)
        
        # Design architecture
        architecture = self.design_architecture(requirements)
        
        # Generate initial code
        code = self.generate_code(requirements, architecture, component_name)
        
        # Apply style guide
        styled_code = self.apply_code_style(code, language, style_guide)
        
        # Format code
        formatted_code = self.format_code(styled_code, language)
        
        # Review code and get improvements
        code_review = self.review_code(formatted_code, requirements, language)
        
        # Write tests
        tests = self.write_tests(formatted_code, requirements, language)
        
        # Write documentation
        documentation = self.write_documentation(formatted_code, architecture, language)
        
        # Return all artifacts
        return {
            "requirements": requirements,
            "architecture": architecture,
            "code": {
                "initial": code["initial_code"],
                "styled": styled_code["styled_code"],
                "formatted": formatted_code["formatted_code"],
                "final": code_review["code_review"]
            },
            "review": code_review["code_review"],
            "tests": tests["tests"],
            "documentation": documentation["documentation"]
        }

# Example usage
if __name__ == "__main__":
    # Set up environment variables for API keys
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")

    # Create the pipeline
    pipeline = CodeDevelopmentPipeline()

    # Define parameters for code development
    project_description = """
    Create a weather forecast application that:
    1. Fetches weather data from a public API
    2. Displays current conditions and 5-day forecast
    3. Allows users to save favorite locations
    4. Sends notifications for severe weather alerts
    5. Works on both mobile and desktop browsers
    """

    component_name = "WeatherDataService"
    language = "python"
    style_guide = "PEP 8"

    # Develop the component
    print("Developing the WeatherDataService component...")
    result = pipeline.develop_component(
        project_description,
        component_name,
        language,
        style_guide
    )

    # Print the final code
    print("\n\n=== FINAL CODE ===\n\n")
    print(result["code"]["final"])

    # Print the tests
    print("\n\n=== TESTS ===\n\n")
    print(result["tests"])
