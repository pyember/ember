"""Multi-agent examples using Ember with Swarm."""

from typing import Any, Dict

from swarm import Agent, Swarm

from ember.integrations.swarm import EmberSwarmClient

# Context variables for shared state
context_variables = {
    "user_name": "",
    "user_preferences": {},
    "conversation_history": [],
    "order_details": {},
}


def transfer_to_sales():
    """Transfer to sales agent."""
    return sales_agent


def transfer_to_support():
    """Transfer to technical support."""
    return support_agent


def transfer_to_manager():
    """Transfer to manager for escalation."""
    return manager_agent


def check_inventory(product_name: str, context_variables: Dict[str, Any]) -> str:
    """Check product inventory."""
    # Simulated inventory check
    inventory = {
        "laptop": {"available": 15, "price": 999.99},
        "phone": {"available": 42, "price": 799.99},
        "tablet": {"available": 8, "price": 499.99},
        "headphones": {"available": 0, "price": 199.99},
    }

    product = product_name.lower()
    if product in inventory:
        item = inventory[product]
        if item["available"] > 0:
            return f"Yes, we have {item['available']} {product}s in stock at ${item['price']} each."
        else:
            return f"Sorry, {product}s are currently out of stock."
    else:
        return f"I couldn't find {product_name} in our inventory."


def place_order(product_name: str, quantity: int, context_variables: Dict[str, Any]) -> str:
    """Place an order for a product."""
    # Simulated order placement
    order_id = f"ORD-{len(context_variables.get('conversation_history', [])) + 1000}"

    context_variables["order_details"] = {
        "order_id": order_id,
        "product": product_name,
        "quantity": quantity,
        "status": "confirmed",
        "customer": context_variables.get("user_name", "Guest"),
    }

    return f"Order {order_id} has been placed for {quantity} {product_name}(s). You'll receive a confirmation email shortly."


def check_order_status(order_id: str, context_variables: Dict[str, Any]) -> str:
    """Check the status of an order."""
    current_order = context_variables.get("order_details", {})

    if current_order.get("order_id") == order_id:
        return f"Order {order_id} is {current_order['status']}. Product: {current_order['product']}, Quantity: {current_order['quantity']}"
    else:
        return f"I couldn't find order {order_id} in our recent records."


def save_user_preference(
    preference_key: str, preference_value: str, context_variables: Dict[str, Any]
) -> str:
    """Save user preferences."""
    if "user_preferences" not in context_variables:
        context_variables["user_preferences"] = {}

    context_variables["user_preferences"][preference_key] = preference_value
    return f"I've saved your preference: {preference_key} = {preference_value}"


# Define agents with different models and capabilities
triage_agent = Agent(
    name="Triage Assistant",
    model="gpt-3.5-turbo",  # Fast, cost-effective for initial routing
    instructions="""You are a friendly triage assistant. Your job is to:
    1. Greet the customer warmly
    2. Understand their needs
    3. Route them to the appropriate specialist:
       - Sales questions → transfer_to_sales()
       - Technical issues → transfer_to_support()
       - Complaints/escalations → transfer_to_manager()
    
    Always be polite and efficient in routing customers.""",
    functions=[transfer_to_sales, transfer_to_support, transfer_to_manager],
)

sales_agent = Agent(
    name="Sales Specialist",
    model="claude-3-haiku-20240307",  # Good balance of capability and cost
    instructions="""You are an enthusiastic sales specialist. You can:
    1. Check product inventory
    2. Provide product recommendations
    3. Process orders
    4. Answer pricing questions
    
    Be helpful and try to find products that match customer needs.
    If there are technical issues, transfer to support.""",
    functions=[check_inventory, place_order, transfer_to_support, save_user_preference],
)

support_agent = Agent(
    name="Technical Support",
    model="gpt-4",  # More capable for complex technical issues
    instructions="""You are a knowledgeable technical support specialist. You can:
    1. Troubleshoot technical issues
    2. Check order status
    3. Provide detailed product specifications
    4. Escalate complex issues to management
    
    Be patient and thorough in resolving customer issues.""",
    functions=[check_order_status, transfer_to_manager, transfer_to_sales],
)

manager_agent = Agent(
    name="Customer Success Manager",
    model="claude-3-opus-20240229",  # Most capable for complex situations
    instructions="""You are a senior customer success manager. You handle:
    1. Escalated complaints
    2. Special requests
    3. Complex problem resolution
    4. Customer retention
    
    You have authority to offer special discounts or solutions.
    Always aim for customer satisfaction while protecting company interests.""",
    functions=[check_order_status, place_order, save_user_preference],
)


def example_customer_journey():
    """Demonstrate a complete customer journey through multiple agents."""
    print("=== Multi-Agent Customer Service Example ===\n")

    # Initialize Ember-backed Swarm
    ember_client = EmberSwarmClient(default_model="gpt-3.5-turbo")
    client = Swarm(client=ember_client)

    # Customer messages simulating a journey
    messages = [
        {"role": "user", "content": "Hi, I'm interested in buying a new laptop"},
        {"role": "user", "content": "What laptops do you have in stock?"},
        {"role": "user", "content": "I'll take one laptop please"},
        {
            "role": "user",
            "content": "Actually, I'm having issues with my previous order. Can you help?",
        },
        {"role": "user", "content": "The screen on my tablet isn't working properly"},
        {
            "role": "user",
            "content": "This is the third time I've had issues. I want to speak to a manager!",
        },
    ]

    # Start with triage agent
    current_agent = triage_agent
    context_variables["user_name"] = "John Doe"
    conversation_messages = []

    print(f"Starting conversation with {current_agent.name}\n")

    for user_message in messages:
        print(f"Customer: {user_message['content']}")
        conversation_messages.append(user_message)

        # Get response from current agent
        response = client.run(
            agent=current_agent,
            messages=conversation_messages,
            context_variables=context_variables,
        )

        # Add assistant response to conversation
        assistant_message = {
            "role": "assistant",
            "content": response.messages[-1]["content"],
        }
        conversation_messages.append(assistant_message)

        # Check if agent changed
        if response.agent != current_agent:
            print(f"\n[Transferred from {current_agent.name} to {response.agent.name}]")
            current_agent = response.agent

        print(f"{current_agent.name}: {response.messages[-1]['content']}\n")

        # Update context
        context_variables.update(response.context_variables)

    # Print final context state
    print("\n=== Final Context State ===")
    print(f"User: {context_variables['user_name']}")
    print(f"Preferences: {context_variables['user_preferences']}")
    print(f"Order Details: {context_variables['order_details']}")

    # Print usage statistics
    print("\n=== Usage Statistics ===")
    stats = ember_client.get_usage_stats()
    print(f"Models used: {', '.join(stats['models_used'])}")
    print(f"Total API calls: {stats.get('total_calls', 'N/A')}")


def example_parallel_agents():
    """Demonstrate parallel agent execution for research tasks."""
    print("=== Parallel Research Agents Example ===\n")

    # Create research agents with different expertise
    market_researcher = Agent(
        name="Market Research Analyst",
        model="claude-3-haiku-20240307",
        instructions="""You are a market research analyst. 
        Analyze market trends, competition, and opportunities.
        Provide data-driven insights and recommendations.""",
    )

    tech_researcher = Agent(
        name="Technical Research Specialist",
        model="gpt-4",
        instructions="""You are a technical research specialist.
        Analyze technical feasibility, requirements, and challenges.
        Provide detailed technical assessments.""",
    )

    finance_researcher = Agent(
        name="Financial Analyst",
        model="claude-3-sonnet-20240229",
        instructions="""You are a financial analyst.
        Analyze costs, ROI, and financial implications.
        Provide budgets and financial projections.""",
    )

    # Initialize client
    ember_client = EmberSwarmClient()
    client = Swarm(client=ember_client)

    # Research topic
    research_topic = "Launching an AI-powered customer service chatbot"

    # Run parallel research
    agents = [market_researcher, tech_researcher, finance_researcher]
    results = {}

    for agent in agents:
        print(f"Consulting {agent.name}...")

        response = client.run(
            agent=agent,
            messages=[
                {
                    "role": "user",
                    "content": f"Please analyze the following business idea and provide insights from your perspective: {research_topic}",
                }
            ],
        )

        results[agent.name] = response.messages[-1]["content"]
        print(f"✓ {agent.name} completed\n")

    # Synthesize results with a coordinator agent
    coordinator = Agent(
        name="Research Coordinator",
        model="claude-3-opus-20240229",
        instructions="""You are a research coordinator.
        Synthesize insights from multiple specialists into a cohesive recommendation.
        Highlight key findings, potential challenges, and actionable next steps.""",
    )

    synthesis_prompt = f"""Please synthesize the following research findings about "{research_topic}":

Market Research:
{results["Market Research Analyst"]}

Technical Research:
{results["Technical Research Specialist"]}

Financial Analysis:
{results["Financial Analyst"]}

Provide a comprehensive summary with recommendations."""

    print("Synthesizing research findings...")
    final_response = client.run(
        agent=coordinator, messages=[{"role": "user", "content": synthesis_prompt}]
    )

    print("\n=== Synthesized Research Report ===")
    print(final_response.messages[-1]["content"])


def example_model_selection_strategy():
    """Demonstrate strategic model selection based on task complexity."""
    print("=== Strategic Model Selection Example ===\n")

    # Define a dynamic agent that selects models based on task
    class DynamicAgent:
        def __init__(self, ember_client: EmberSwarmClient):
            self.ember_client = ember_client
            self.task_patterns = {
                "simple": "gpt-3.5-turbo",
                "moderate": "claude-3-haiku-20240307",
                "complex": "gpt-4",
                "creative": "claude-3-opus-20240229",
            }

        def classify_task_complexity(self, message: str) -> str:
            """Simple heuristic for task classification."""
            word_count = len(message.split())

            if any(
                word in message.lower() for word in ["analyze", "compare", "evaluate", "strategy"]
            ):
                return "complex"
            elif any(word in message.lower() for word in ["create", "write", "design", "imagine"]):
                return "creative"
            elif word_count > 50 or "?" in message:
                return "moderate"
            else:
                return "simple"

        def create_agent_for_task(self, task: str) -> Agent:
            """Create an agent with appropriate model for the task."""
            complexity = self.classify_task_complexity(task)
            model = self.task_patterns[complexity]

            return Agent(
                name=f"Dynamic Assistant ({complexity})",
                model=model,
                instructions="You are a helpful assistant. Provide clear, accurate responses.",
            )

    # Initialize
    ember_client = EmberSwarmClient()
    client = Swarm(client=ember_client)
    dynamic_system = DynamicAgent(ember_client)

    # Test various tasks
    tasks = [
        "What's the weather like?",
        "Explain how neural networks work in simple terms",
        "Create a haiku about artificial intelligence",
        "Analyze the economic impact of remote work on urban real estate markets and provide strategic recommendations",
    ]

    for task in tasks:
        # Create appropriate agent
        agent = dynamic_system.create_agent_for_task(task)

        print(f"Task: {task}")
        print(f"Selected Model: {agent.model}")

        # Get response
        response = client.run(agent=agent, messages=[{"role": "user", "content": task}])

        print(f"Response: {response.messages[-1]['content'][:200]}...")
        print("-" * 60 + "\n")

    # Show cost optimization
    print("=== Cost Optimization Summary ===")
    stats = ember_client.get_usage_stats()
    print(f"Models used: {stats['models_used']}")
    print("By selecting appropriate models for each task, we optimize for both quality and cost.")


if __name__ == "__main__":
    # Run examples
    example_customer_journey()
    print("\n" + "=" * 80 + "\n")

    example_parallel_agents()
    print("\n" + "=" * 80 + "\n")

    example_model_selection_strategy()
