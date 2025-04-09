"""Multi-Agent Content Creation Studio.

This module demonstrates a multi-agent system for creating content.
"""

import logging
from typing import Any, Dict, List, Optional

from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentCreationStudio:
    """A multi-agent system for collaborative content creation.
    
    This studio uses specialized agents to handle different aspects of content creation:
    - Content Planner
    - Researcher
    - Content Drafter
    - Editor
    - Style Specialist
    - Formatter
    """

    def __init__(self):
        """Initialize the specialized agents with different model configurations."""
        # All agents using OpenAI models with different temperatures for specialization
        self.content_planner = models.model("openai:gpt-4-turbo", temperature=0.7)  # More creative for ideation
        self.researcher = models.model("openai:gpt-4-turbo", temperature=0.2)  # More factual for research
        self.content_drafter = models.model("openai:gpt-4-turbo", temperature=0.5)  # Balanced for drafting
        self.editor = models.model("openai:gpt-4-turbo", temperature=0.3)  # More critical for editing
        self.style_specialist = models.model("openai:gpt-4-turbo", temperature=0.4)  # Creative but controlled for style
        self.formatter = models.model("openai:gpt-4-turbo", temperature=0.1)  # More precise for formatting

    def create_content_plan(self, topic: str, target_audience: str, content_type: str) -> Dict[str, Any]:
        """Create a content plan based on topic, audience and content type.
        
        Args:
            topic: The main topic of the content
            target_audience: Description of the target audience
            content_type: Type of content (e.g., blog post, social media, whitepaper)
            
        Returns:
            Dictionary containing the content plan
        """
        logger.info(f"Creating content plan for {content_type} on {topic}...")
        prompt = f"""Create a detailed content plan for a {content_type} about {topic} aimed at {target_audience}.

Your plan should include:
1. A compelling headline/title
2. Content objectives and key messages
3. Detailed outline with section headings
4. Key points to cover in each section
5. Recommended tone and style
6. Suggested content length

Format your response as JSON with these categories.
"""
        result = self.content_planner(prompt)
        return {"content_plan": result}

    def conduct_research(self, topic: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct research on the topic based on the content plan.
        
        Args:
            topic: The main topic to research
            plan: Dictionary containing the content plan
            
        Returns:
            Dictionary containing research findings
        """
        logger.info(f"Conducting research on {topic}...")
        prompt = f"""Conduct thorough research on {topic} to support this content plan:

{plan["content_plan"]}

Your research should:
1. Identify key facts, statistics, and data points
2. Find relevant examples and case studies
3. Identify expert opinions and quotations (with attributions)
4. Uncover common questions and misconceptions
5. Find trending topics related to {topic}

Format your response as JSON with these categories and include sources where applicable.
"""
        result = self.researcher(prompt)
        return {"research": result}

    def generate_draft(self, plan: Dict[str, Any], research: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a content draft based on the plan and research.
        
        Args:
            plan: Dictionary containing the content plan
            research: Dictionary containing research findings
            
        Returns:
            Dictionary containing the draft content
        """
        logger.info("Generating content draft...")
        prompt = f"""Create a comprehensive first draft based on this content plan and research:

Content Plan:
{plan["content_plan"]}

Research:
{research["research"]}

Write a complete draft that:
1. Follows the structure in the content plan
2. Incorporates key facts and insights from the research
3. Uses a conversational but informative tone
4. Includes an engaging introduction and conclusion
5. Weaves in examples and data points naturally

Return only the draft content with no additional commentary.
"""
        result = self.content_drafter(prompt)
        return {"draft": result}

    def edit_content(self, draft: Dict[str, Any]) -> Dict[str, Any]:
        """Edit the draft content for clarity, flow, and accuracy.
        
        Args:
            draft: Dictionary containing the draft content
            
        Returns:
            Dictionary containing the edited content
        """
        logger.info("Editing content...")
        prompt = f"""Edit this content draft for clarity, flow, and impact:

{draft["draft"]}

Improve the content by:
1. Enhancing clarity and readability
2. Improving paragraph and sentence structure
3. Eliminating redundancies and filler content
4. Strengthening transitions between sections
5. Ensuring logical flow of ideas
6. Maintaining a consistent voice throughout

Return only the edited content with no additional commentary.
"""
        result = self.editor(prompt)
        return {"edited_content": result}

    def apply_style(self, edited_content: Dict[str, Any], style_guide: str) -> Dict[str, Any]:
        """Apply the specified style guide to the edited content.
        
        Args:
            edited_content: Dictionary containing the edited content
            style_guide: Style guide to apply
            
        Returns:
            Dictionary containing the styled content
        """
        logger.info("Applying style guidelines...")
        prompt = f"""Apply the following style guide to this edited content:

Style Guide:
{style_guide}

Content:
{edited_content["edited_content"]}

Apply the style by:
1. Adjusting the tone and voice to match the style guide
2. Using appropriate terminology and phrasing
3. Applying brand language elements where relevant
4. Ensuring sentence structure aligns with the style
5. Maintaining consistency with the specified style throughout

Return only the styled content with no additional commentary.
"""
        result = self.style_specialist(prompt)
        return {"styled_content": result}

    def format_content(self, styled_content: Dict[str, Any], format_requirements: str, platform: str) -> Dict[str, Any]:
        """Format the styled content according to specified requirements and platform.
        
        Args:
            styled_content: Dictionary containing the styled content
            format_requirements: Formatting requirements specification
            platform: Target platform (e.g., WordPress, Medium, LinkedIn)
            
        Returns:
            Dictionary containing the formatted content
        """
        logger.info(f"Formatting content for {platform}...")
        prompt = f"""Format this content according to these requirements for {platform}:

Format Requirements:
{format_requirements}

Content:
{styled_content["styled_content"]}

Apply formatting by:
1. Structuring the content with proper headings and subheadings
2. Adding appropriate formatting tags/markdown for {platform}
3. Breaking up text with lists, callouts, and spacing as needed
4. Including any platform-specific elements
5. Ensuring the format enhances readability and engagement

Return the formatted content in a format appropriate for {platform}.
"""
        result = self.formatter(prompt)
        return {"final_formatted": result}

    def create_complete_content(self, topic: str, target_audience: str, content_type: str, style_guide: str, format_requirements: str, platform: str) -> Dict[str, Any]:
        """Execute the full content creation pipeline.
        
        Args:
            topic: The main topic of the content
            target_audience: Description of the target audience
            content_type: Type of content (e.g., blog post, social media, whitepaper)
            style_guide: Style guide to apply
            format_requirements: Formatting requirements specification
            platform: Target platform (e.g., WordPress, Medium, LinkedIn)
            
        Returns:
            Dictionary with all artifacts from the content creation process
        """
        # Create content plan
        plan = self.create_content_plan(topic, target_audience, content_type)
        
        # Conduct research
        research = self.conduct_research(topic, plan)
        
        # Generate draft
        draft = self.generate_draft(plan, research)
        
        # Edit content
        edited_content = self.edit_content(draft)
        
        # Apply style
        styled_content = self.apply_style(edited_content, style_guide)
        
        # Format content
        formatted_content = self.format_content(styled_content, format_requirements, platform)
        
        # Return all artifacts
        return {
            "plan": plan,
            "research": research,
            "draft": draft["draft"],
            "edited_content": edited_content["edited_content"],
            "styled_content": styled_content["styled_content"],
            "final_formatted": formatted_content["final_formatted"]
        }

# Example usage
if __name__ == "__main__":
    # Set up environment variables for API keys
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")

    # Create the studio
    studio = ContentCreationStudio()

    # Define parameters for content creation
    topic = "Sustainable Urban Gardening"
    target_audience = "Urban millennials interested in sustainability and home gardening"
    content_type = "blog post"

    style_guide = """
    Voice: Friendly, informative, and encouraging
    Tone: Conversational but authoritative
    Terminology: Use accessible gardening terms, explain technical concepts
    Sentence structure: Mix of short and medium-length sentences, avoid complex structures
    Brand elements: Emphasize sustainability, community, and practical solutions
    """

    format_requirements = """
    - 1500-2000 words
    - H1 main title
    - H2 for main sections
    - H3 for subsections
    - Short paragraphs (3-4 sentences max)
    - Include bulleted lists where appropriate
    - Bold key points and important terms
    - Include a "Quick Tips" section
    - End with a call-to-action
    """

    platform = "WordPress blog"

    # Generate the content
    print("Generating complete content for 'Sustainable Urban Gardening'...")
    content = studio.create_complete_content(
        topic,
        target_audience,
        content_type,
        style_guide,
        format_requirements,
        platform
    )

    # Print the final formatted content
    print("\n\n=== FINAL FORMATTED CONTENT ===\n\n")
    print(content["final_formatted"])
