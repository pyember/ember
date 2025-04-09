# Multi-Agent Systems with Ember

This repository contains implementations of various multi-agent systems using the Ember framework. Each system demonstrates how specialized agents can work together to accomplish complex tasks, with dedicated styling and formatting agents to ensure high-quality output.

## Overview

All systems use OpenAI models exclusively, with different temperature settings to specialize each agent for its specific role. The systems demonstrate how multiple agents can collaborate on complex tasks, with each agent focusing on a specific aspect of the task.

## Multi-Agent Systems

### 1. Content Creation Studio

A multi-agent system for creating high-quality content, from planning to final formatting.

**Agents:**
- **Content Planner**: Creates detailed content plans
- **Researcher**: Conducts comprehensive research on topics
- **Content Creator**: Generates initial drafts based on research
- **Editor**: Improves clarity, coherence, and impact
- **Stylist**: Ensures adherence to style guidelines
- **Formatter**: Optimizes formatting for different platforms

**File:** `content_creation_studio.py`

### 2. Code Development Pipeline

A multi-agent system for developing software components, from requirements analysis to documentation.

**Agents:**
- **Requirements Analyzer**: Formalizes project requirements
- **Architect**: Designs system architecture
- **Code Generator**: Writes code based on specifications
- **Code Stylist**: Ensures code follows style guidelines
- **Code Formatter**: Handles proper formatting and organization
- **Code Reviewer**: Reviews code for quality and security
- **Test Writer**: Creates comprehensive tests
- **Documentation Writer**: Creates technical documentation

**File:** `code_development_pipeline.py`

### 3. Educational Content Generator

A multi-agent system for creating educational materials tailored to different learning styles and needs.

**Agents:**
- **Curriculum Designer**: Designs comprehensive curricula
- **Subject Expert**: Creates expert-level content
- **Content Creator**: Develops engaging learning materials
- **Assessment Designer**: Creates comprehensive assessments
- **Educational Stylist**: Applies pedagogical approaches
- **Content Formatter**: Formats content for delivery platforms
- **Accessibility Specialist**: Ensures materials are accessible

**File:** `educational_content_generator.py`

### 4. Marketing Campaign Generator

A multi-agent system for creating comprehensive marketing campaigns.

**Agents:**
- **Market Researcher**: Analyzes target audience and competitors
- **Campaign Strategist**: Develops marketing strategies
- **Content Creator**: Creates campaign content
- **Copywriter**: Writes marketing copy
- **Brand Stylist**: Ensures adherence to brand guidelines
- **Channel Formatter**: Optimizes content for different channels
- **Campaign Analyzer**: Analyzes campaign effectiveness

**File:** `marketing_campaign_generator.py`

## Running the Systems

You can run all systems or individual systems using the provided script:

```bash
# Run all systems
python run_multi_agent_systems.py all

# Run individual systems
python run_multi_agent_systems.py content
python run_multi_agent_systems.py code
python run_multi_agent_systems.py education
python run_multi_agent_systems.py marketing
```

## Requirements

- Ember framework
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Implementation Details

Each multi-agent system follows a similar pattern:

1. **Specialized Agents**: Each agent is implemented as an instance of a model with specific temperature settings to optimize for its role.
2. **Sequential Processing**: Agents work in sequence, with each agent building on the output of previous agents.
3. **Styling and Formatting**: Dedicated styling and formatting agents ensure the output adheres to guidelines and is properly formatted.
4. **Quality Control**: Review and analysis agents ensure the quality of the final output.

## Example Outputs

Each system produces high-quality outputs in its domain:

- **Content Creation Studio**: Well-structured, styled, and formatted blog posts
- **Code Development Pipeline**: Clean, well-documented code with comprehensive tests
- **Educational Content Generator**: Engaging, accessible educational materials
- **Marketing Campaign Generator**: Effective marketing content optimized for different channels

## Future Enhancements

Potential enhancements for these multi-agent systems include:

1. **Parallel Processing**: Allow agents to work in parallel when possible
2. **Feedback Loops**: Implement feedback loops between agents
3. **User Interaction**: Add capabilities for user feedback during the process
4. **Additional Specializations**: Add more specialized agents for specific tasks
5. **Cross-System Integration**: Enable different systems to work together
