"""Educational Content Generator.

This module demonstrates a multi-agent system for creating educational content.
"""

import logging
from typing import Any, Dict, List, Optional

from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EducationalContentGenerator:
    def __init__(self):
        # All agents using OpenAI models with different specializations
        self.curriculum_designer = models.model("openai:gpt-4o", temperature=0.4)
        self.subject_expert = models.model("openai:gpt-4o", temperature=0.3)
        self.content_creator = models.model("openai:gpt-4o", temperature=0.6)
        self.assessment_designer = models.model("openai:gpt-4o", temperature=0.4)
        self.educational_stylist = models.model("openai:gpt-4o", temperature=0.3)  # Specialized for educational styling
        self.content_formatter = models.model("openai:gpt-4o", temperature=0.2)  # Specialized for formatting
        self.accessibility_specialist = models.model("openai:gpt-4o", temperature=0.3)

    def design_curriculum(self, subject, grade_level, learning_objectives):
        curriculum = self.curriculum_designer(
            f"""Design a comprehensive curriculum for {subject} at {grade_level} grade level.

            Learning Objectives:
            {learning_objectives}

            Include:
            - Unit structure and sequence
            - Key concepts for each unit
            - Skill progression throughout the curriculum
            - Suggested timeframes
            - Prerequisites and connections to other subjects
            """
        )
        return curriculum

    def create_expert_content(self, curriculum, unit_name):
        expert_content = self.subject_expert(
            f"""Create expert-level content for the {unit_name} unit in this curriculum:

            {curriculum}

            Provide:
            - Comprehensive explanation of all concepts
            - Accurate and up-to-date information
            - Common misconceptions and clarifications
            - Advanced examples and applications
            - Historical context and future directions
            """
        )
        return expert_content

    def create_learning_materials(self, expert_content, grade_level, learning_styles):
        learning_materials = self.content_creator(
            f"""Create engaging learning materials based on this expert content for {grade_level} grade level:

            Expert Content:
            {expert_content}

            Learning Styles to Address:
            {learning_styles}

            Create:
            - Lesson plans with clear objectives
            - Engaging explanations and examples
            - Interactive activities and exercises
            - Visual aids and diagrams
            - Real-world applications and scenarios
            """
        )
        return learning_materials

    def create_assessments(self, learning_materials, learning_objectives):
        assessments = self.assessment_designer(
            f"""Create comprehensive assessments for these learning materials:

            Learning Materials:
            {learning_materials}

            Learning Objectives:
            {learning_objectives}

            Include:
            - Formative assessments for ongoing feedback
            - Summative assessments for unit completion
            - Various question types (multiple choice, short answer, essay, etc.)
            - Performance tasks and projects
            - Rubrics and scoring guidelines
            """
        )
        return assessments

    def apply_educational_style(self, learning_materials, pedagogical_approach, grade_level):
        styled_materials = self.educational_stylist(
            f"""Apply the {pedagogical_approach} pedagogical approach to these learning materials for {grade_level} grade level:

            Learning Materials:
            {learning_materials}

            Ensure the materials:
            - Use age-appropriate language and examples
            - Follow best practices for the specified pedagogical approach
            - Maintain consistent terminology and presentation
            - Use effective instructional techniques
            - Incorporate appropriate scaffolding
            """
        )
        return styled_materials

    def format_educational_content(self, styled_materials, delivery_format):
        formatted_materials = self.content_formatter(
            f"""Format these educational materials for {delivery_format}:

            Materials:
            {styled_materials}

            Apply formatting appropriate for {delivery_format}, including:
            - Proper headings and structure
            - Visual organization of information
            - Consistent layout and design
            - Appropriate use of space and breaks
            - Format-specific elements and features
            """
        )
        return formatted_materials

    def ensure_accessibility(self, formatted_materials, accessibility_requirements):
        accessible_materials = self.accessibility_specialist(
            f"""Enhance these educational materials to meet these accessibility requirements:

            Materials:
            {formatted_materials}

            Accessibility Requirements:
            {accessibility_requirements}

            Ensure the materials:
            - Are accessible to students with various disabilities
            - Include alternative representations of visual content
            - Use accessible language and structure
            - Can be navigated with assistive technologies
            - Meet specified accessibility standards
            """
        )
        return accessible_materials

    def generate_educational_unit(self, subject, unit_name, grade_level, learning_objectives,
                                 learning_styles, pedagogical_approach, delivery_format,
                                 accessibility_requirements):
        # Design curriculum
        curriculum = self.design_curriculum(subject, grade_level, learning_objectives)
        curriculum_text = str(curriculum)

        # Create expert content
        expert_content = self.create_expert_content(curriculum_text, unit_name)
        expert_content_text = str(expert_content)

        # Create learning materials
        learning_materials = self.create_learning_materials(expert_content_text, grade_level, learning_styles)
        learning_materials_text = str(learning_materials)

        # Create assessments
        assessments = self.create_assessments(learning_materials_text, learning_objectives)
        assessments_text = str(assessments)

        # Apply educational styling
        styled_materials = self.apply_educational_style(learning_materials_text, pedagogical_approach, grade_level)
        styled_materials_text = str(styled_materials)

        # Format content for delivery
        formatted_materials = self.format_educational_content(styled_materials_text, delivery_format)
        formatted_materials_text = str(formatted_materials)

        # Ensure accessibility
        accessible_materials = self.ensure_accessibility(formatted_materials_text, accessibility_requirements)
        accessible_materials_text = str(accessible_materials)

        return {
            "curriculum": curriculum_text,
            "expert_content": expert_content_text,
            "learning_materials": {
                "raw": learning_materials_text,
                "styled": styled_materials_text,
                "formatted": formatted_materials_text,
                "accessible": accessible_materials_text
            },
            "assessments": assessments_text
        }

if __name__ == "__main__":
    # Set up environment variables for API keys
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")
        
    # Create the generator
    generator = EducationalContentGenerator()

    # Define parameters for educational content generation
    subject = "Environmental Science"
    unit_name = "Ecosystems and Biodiversity"
    grade_level = "8th"

    learning_objectives = """
    1. Understand the components and interactions within ecosystems
    2. Explain the importance of biodiversity for ecosystem health
    3. Identify human impacts on ecosystems and biodiversity
    4. Analyze solutions for preserving biodiversity
    5. Design a plan to protect a local ecosystem
    """

    learning_styles = """
    - Visual learners: diagrams, charts, videos
    - Auditory learners: discussions, audio explanations
    - Kinesthetic learners: hands-on activities, experiments
    - Reading/writing learners: text-based materials, note-taking activities
    """

    pedagogical_approach = "inquiry-based learning"
    delivery_format = "digital learning management system"

    accessibility_requirements = """
    - Screen reader compatibility
    - Alternative text for images
    - Transcripts for audio content
    - Color contrast for visually impaired students
    - Multiple means of engagement and expression
    """

    # Generate the educational unit
    print("Generating educational unit for 'Ecosystems and Biodiversity'...")
    unit = generator.generate_educational_unit(
        subject,
        unit_name,
        grade_level,
        learning_objectives,
        learning_styles,
        pedagogical_approach,
        delivery_format,
        accessibility_requirements
    )

    # Print the accessible learning materials
    print("\n\n=== ACCESSIBLE LEARNING MATERIALS ===\n\n")
    print(unit["learning_materials"]["accessible"])
