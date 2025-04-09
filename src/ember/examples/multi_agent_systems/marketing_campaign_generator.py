"""Marketing Campaign Generator.

This module demonstrates a multi-agent system for creating marketing campaigns.
"""

import logging
from typing import Any, Dict, List, Optional

from ember.api import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketingCampaignGenerator:
    def __init__(self):
        # All agents using OpenAI models with different specializations
        self.market_researcher = models.model("openai:gpt-4o", temperature=0.3)
        self.campaign_strategist = models.model("openai:gpt-4o", temperature=0.5)
        self.content_creator = models.model("openai:gpt-4o", temperature=0.7)
        self.copywriter = models.model("openai:gpt-4o", temperature=0.6)
        self.brand_stylist = models.model("openai:gpt-4o", temperature=0.3)  # Specialized for brand styling
        self.channel_formatter = models.model("openai:gpt-4o", temperature=0.2)  # Specialized for channel-specific formatting
        self.campaign_analyzer = models.model("openai:gpt-4o", temperature=0.2)

    def conduct_market_research(self, product, target_audience, competitors):
        research = self.market_researcher(
            f"""Conduct comprehensive market research for {product} targeting {target_audience}.

            Competitors to analyze:
            {competitors}

            Provide:
            - Target audience demographics and psychographics
            - Market trends and opportunities
            - Competitor analysis and positioning
            - Customer pain points and needs
            - Unique selling propositions for the product
            """
        )
        return research

    def develop_campaign_strategy(self, research, campaign_objectives, budget_range):
        strategy = self.campaign_strategist(
            f"""Develop a marketing campaign strategy based on this research:

            Research:
            {research}

            Campaign Objectives:
            {campaign_objectives}

            Budget Range:
            {budget_range}

            Include:
            - Campaign theme and key messages
            - Marketing channels and tactics
            - Budget allocation
            - Timeline and key milestones
            - KPIs and success metrics
            """
        )
        return strategy

    def create_campaign_content(self, strategy, brand_guidelines):
        content = self.content_creator(
            f"""Create marketing campaign content based on this strategy:

            Strategy:
            {strategy}

            Brand Guidelines:
            {brand_guidelines}

            Develop:
            - Campaign tagline and slogans
            - Key messaging points
            - Content themes and concepts
            - Visual direction recommendations
            - Campaign narrative and storytelling elements
            """
        )
        return content

    def write_marketing_copy(self, content, target_audience, channels):
        copy = self.copywriter(
            f"""Write marketing copy based on this campaign content:

            Campaign Content:
            {content}

            Target Audience:
            {target_audience}

            Marketing Channels:
            {channels}

            Create copy for each specified channel that:
            - Resonates with the target audience
            - Communicates key messages effectively
            - Drives desired actions
            - Maintains consistent voice across channels
            - Adapts to channel-specific requirements
            """
        )
        return copy

    def apply_brand_style(self, copy, brand_guidelines):
        styled_copy = self.brand_stylist(
            f"""Apply these brand guidelines to the marketing copy:

            Marketing Copy:
            {copy}

            Brand Guidelines:
            {brand_guidelines}

            Ensure the copy:
            - Adheres to brand voice and tone
            - Uses approved terminology and language
            - Follows brand messaging hierarchy
            - Incorporates brand values and personality
            - Maintains brand consistency
            """
        )
        return styled_copy

    def format_for_channels(self, styled_copy, channels):
        formatted_content = {}

        for channel in channels:
            formatted = self.channel_formatter(
                f"""Format this marketing copy for {channel}:

                Styled Copy:
                {styled_copy}

                Apply formatting specific to {channel}, including:
                - Character/word count limitations
                - Platform-specific features and capabilities
                - Best practices for engagement on this channel
                - Technical requirements and constraints
                - Optimal content structure for this channel
                """
            )
            formatted_content[channel] = formatted

        return formatted_content

    def analyze_campaign_effectiveness(self, strategy, formatted_content):
        analysis = self.campaign_analyzer(
            f"""Analyze the potential effectiveness of this marketing campaign:

            Campaign Strategy:
            {strategy}

            Campaign Content:
            {formatted_content}

            Provide:
            - Strengths and weaknesses of the campaign
            - Alignment with campaign objectives
            - Potential challenges and risks
            - Recommendations for optimization
            - Expected outcomes and impact
            """
        )
        return analysis

    def generate_marketing_campaign(self, product, target_audience, competitors, campaign_objectives,
                                   budget_range, brand_guidelines, channels):
        # Conduct market research
        research = self.conduct_market_research(product, target_audience, competitors)
        research_text = str(research)

        # Develop campaign strategy
        strategy = self.develop_campaign_strategy(research_text, campaign_objectives, budget_range)
        strategy_text = str(strategy)

        # Create campaign content
        content = self.create_campaign_content(strategy_text, brand_guidelines)
        content_text = str(content)

        # Write marketing copy
        copy = self.write_marketing_copy(content_text, target_audience, channels)
        copy_text = str(copy)

        # Apply brand styling
        styled_copy = self.apply_brand_style(copy_text, brand_guidelines)
        styled_copy_text = str(styled_copy)

        # Format for different channels
        formatted_content = {}
        for channel in channels:
            channel_content = self.channel_formatter(
                f"""Format this marketing copy for {channel}:

                Styled Copy:
                {styled_copy_text}

                Apply formatting specific to {channel}, including:
                - Character/word count limitations
                - Platform-specific features and capabilities
                - Best practices for engagement on this channel
                - Technical requirements and constraints
                - Optimal content structure for this channel
                """
            )
            formatted_content[channel] = str(channel_content)

        # Analyze campaign effectiveness
        analysis = self.analyze_campaign_effectiveness(strategy_text, str(formatted_content))
        analysis_text = str(analysis)

        return {
            "research": research_text,
            "strategy": strategy_text,
            "content": content_text,
            "copy": copy_text,
            "styled_copy": styled_copy_text,
            "formatted_content": formatted_content,
            "analysis": analysis_text
        }

if __name__ == "__main__":
    # Set up environment variables for API keys
    import os
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")
        
    # Create the generator
    generator = MarketingCampaignGenerator()

    # Define parameters for marketing campaign generation
    product = "EcoCharge - Solar-powered portable charger for mobile devices"

    target_audience = "Environmentally conscious millennials and Gen Z who enjoy outdoor activities"

    competitors = """
    1. SolarJuice - Premium solar chargers with high price point
    2. PowerGreen - Budget solar chargers with lower efficiency
    3. NaturePower - Well-established brand with wide product range
    """

    campaign_objectives = """
    1. Increase brand awareness by 30% among target audience
    2. Generate 10,000 website visits within the first month
    3. Achieve 2,000 product sales in the first quarter
    4. Establish EcoCharge as an eco-friendly tech leader
    """

    budget_range = "$50,000 - $75,000"

    brand_guidelines = """
    Voice: Friendly, enthusiastic, and environmentally conscious
    Tone: Inspirational, educational, and slightly playful
    Colors: Green (#2E8B57), Blue (#1E90FF), White (#FFFFFF)
    Typography: Clean, modern sans-serif fonts
    Values: Sustainability, innovation, quality, adventure
    Messaging: Focus on environmental impact, convenience, and reliability
    """

    channels = ["Instagram", "TikTok", "Email Newsletter", "Google Ads", "Outdoor Retailer Partnerships"]

    # Generate the marketing campaign
    print("Generating marketing campaign for 'EcoCharge'...")
    campaign = generator.generate_marketing_campaign(
        product,
        target_audience,
        competitors,
        campaign_objectives,
        budget_range,
        brand_guidelines,
        channels
    )

    # Print the formatted content for Instagram
    print("\n\n=== INSTAGRAM CONTENT ===\n\n")
    print(campaign["formatted_content"]["Instagram"])
