# # Summoned via the ModelService with an ENUM
# response = model_service(ModelEnum.OPENAI_GPT4, "Hello world!")
# print(response.data)

# # Or: direct usage
# gpt4_model = model_service.get_model(ModelEnum.OPENAI_GPT4)
# response = gpt4_model("What is the capital of France?")
# print(response.data)

# model_service = ModelService(registry)

# # Option 1: Using enum
# response = model_service(ModelEnum.OPENAI_GPT4, "What is the capital of France?")
# print(response.data)
# print(response.usage)

# # Option 2: Using string
# response = model_service("openai:gpt-4", "What is the capital of France?")

# # Pytorch-like
# gpt4_model = model_service.get_model(ModelEnum.OPENAI_GPT4)
# response_via_forward = gpt4_model.forward("What is the capital of France?")
# response_via_call = gpt4_model("What is the capital of France?")
# print(response_via_forward.data)
# print(response_via_call.data)