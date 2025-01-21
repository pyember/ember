# Summoned via the ModelService with an ENUM
response = model_service(ModelEnum.OPENAI_GPT4, "Hello world!")
print(response.data)

# Or: direct usage
gpt4_model = model_service.get_model(ModelEnum.OPENAI_GPT4)
response = gpt4_model("What is the capital of France?")
print(response.data)

model_service = ModelService(registry)

# Option 1: Using enum
response = model_service(ModelEnum.OPENAI_GPT4, "What is the capital of France?")
print(response.data)
print(response.usage)

# Option 2: Using string
response = model_service("openai:gpt-4", "What is the capital of France?")

# Pytorch-like
gpt4_model = model_service.get_model(ModelEnum.OPENAI_GPT4)
response = gpt4_model.forward("What is the capital of France?")
print(response.data)

class CaravanLabelingInputs(BaseModel):
    question: str

class CaravanLabelingSignature(Signature):
    """Signature for labeling network flows as benign or malicious."""

    required_inputs: List[str] = ["question"]
    prompt_template: str = (
        "You are a network security expert.\n"
        "Given these unlabeled flows:\n{question}\n"
        "Label each flow as 0 for benign or 1 for malicious, one per line, no explanation.\n"
    )
    structured_output: Optional[Type[BaseModel]] = CaravanLabelsOutput
    input_model: Optional[Type[BaseModel]] = CaravanLabelingInputs

Example usage with an ensemble operator:
from avior.registry.operators.operator_registry import EnsembleOperator, EnsembleOperatorInputs
from avior.registry.operators.operator_base import LMModuleConfig, LMModule

question_data = "What is the capital of France?"

# Define LMModule configurations
lm_configs = [
    LMModuleConfig(model_name="gpt-4o", temperature=0.7),
    LMModuleConfig(model_name="gpt-4o-mini", temperature=0.7),
    LMModuleConfig(model_name="gpt-4o-2024-08-06", temperature=0.7),
]

# Create LMModules from the configurations
lm_modules = [LMModule(config=config) for config in lm_configs]

# Create an instance of the EnsembleOperator with the LMModules and signature
caravan_operator = EnsembleOperator(
    lm_modules=lm_modules,
)

# Build the inputs using the signature's input model
inputs = caravan_operator.build_inputs(query=question_data)

# Execute the operator
result = caravan_operator(inputs=inputs)
print(result)
