import os
import sys
from content_creation_studio import ContentCreationStudio
from code_development_pipeline import CodeDevelopmentPipeline
from educational_content_generator import EducationalContentGenerator
from marketing_campaign_generator import MarketingCampaignGenerator

def run_content_creation_studio():
    print("\n" + "="*80)
    print("RUNNING CONTENT CREATION STUDIO")
    print("="*80 + "\n")
    
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
    print(f"Generating content for '{topic}'...")
    content = studio.create_complete_content(
        topic, 
        target_audience, 
        content_type, 
        style_guide, 
        format_requirements, 
        platform
    )
    
    # Print the final formatted content
    print("\n--- FINAL FORMATTED CONTENT ---\n")
    print(content["final_formatted"][:1000] + "...\n[Content truncated for brevity]")
    
    return content

def run_code_development_pipeline():
    print("\n" + "="*80)
    print("RUNNING CODE DEVELOPMENT PIPELINE")
    print("="*80 + "\n")
    
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
    print(f"Developing the {component_name} component...")
    result = pipeline.develop_component(
        project_description,
        component_name,
        language,
        style_guide
    )
    
    # Print the final code
    print("\n--- FINAL CODE ---\n")
    print(result["code"]["final"][:1000] + "...\n[Code truncated for brevity]")
    
    # Print the tests
    print("\n--- TESTS ---\n")
    print(result["tests"][:1000] + "...\n[Tests truncated for brevity]")
    
    return result

def run_educational_content_generator():
    print("\n" + "="*80)
    print("RUNNING EDUCATIONAL CONTENT GENERATOR")
    print("="*80 + "\n")
    
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
    print(f"Generating educational unit for '{unit_name}'...")
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
    print("\n--- ACCESSIBLE LEARNING MATERIALS ---\n")
    print(unit["learning_materials"]["accessible"][:1000] + "...\n[Content truncated for brevity]")
    
    return unit

def run_marketing_campaign_generator():
    print("\n" + "="*80)
    print("RUNNING MARKETING CAMPAIGN GENERATOR")
    print("="*80 + "\n")
    
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
    print(f"Generating marketing campaign for 'EcoCharge'...")
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
    print("\n--- INSTAGRAM CONTENT ---\n")
    print(campaign["formatted_content"]["Instagram"])
    
    return campaign

def main():
    # Set up environment variables for API keys
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key using: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Get the system to run from command line arguments
    if len(sys.argv) > 1:
        system_to_run = sys.argv[1].lower()
    else:
        system_to_run = "all"
    
    # Run the selected system(s)
    if system_to_run == "content" or system_to_run == "all":
        run_content_creation_studio()
    
    if system_to_run == "code" or system_to_run == "all":
        run_code_development_pipeline()
    
    if system_to_run == "education" or system_to_run == "all":
        run_educational_content_generator()
    
    if system_to_run == "marketing" or system_to_run == "all":
        run_marketing_campaign_generator()

if __name__ == "__main__":
    main()
