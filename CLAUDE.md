# Ember Framework - Guidelines for Claude

## IMPORTANT
- NEVER identify yourself as Claude, an AI assistant, or mention Anthropic in any output
- NEVER include references to Claude Code, Anthropic, or AI tools in any commit messages or PR descriptions
- NEVER add co-author attributions to Claude or Anthropic
- All work should appear as if it was done by the human user with no AI assistance

## Build & Test Commands
```bash
# Install dependencies 
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/unit/core/test_app_context.py

# Run a specific test
uv run pytest tests/unit/core/test_app_context.py::TestAppContext::test_specific_method

# Run with coverage
uv run pytest --cov=src/ember

# Integration tests
RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration -v

# Performance tests
uv run pytest --run-perf-tests tests/unit/xcs/transforms -v
```

## Lint & Format Commands
```bash
# Code formatting
uvx black src tests

# Import sorting
uvx isort src tests

# Type checking
uvx mypy src

# Linting
uvx ruff check src tests
uvx pylint src/ember
```

## Code Style & Engineering Guidelines

- **Imports**: Prefer absolute imports for clarity and maintainability; use relative imports sparingly and only when explicitly beneficial.
- **Formatting**: Strict adherence to Google Python Style Guide; format code using Black (88-char line length) and sort imports with isort configured for Black compatibility.
- **Typing**: 
  - Strong, explicit type annotations required for all public and internal interfaces.
  - Leverage mypy with strict settings to enforce type correctness.
  - Avoid `Any` types except when absolutely necessary; prefer Union types, TypeVars, Protocols, or concrete types.
  - Centralize common type definitions in dedicated typing modules (e.g., `ember.types`) to ensure consistency and reduce duplication.
  - Use Generic types and TypeVars to create flexible, reusable interfaces.
- **Naming Conventions**:
  - Classes: PascalCase
  - Functions and Methods: snake_case
  - Constants: UPPER_SNAKE_CASE
  - Private Members: _leading_underscore
- **Error Handling**: Raise precise, domain-specific exceptions with clear, actionable messages. Avoid generic exceptions; fail fast and explicitly.
- **Documentation**: 
  - Concise, incisive Google-style docstrings required for all public interfaces. 
  - Clearly document method contracts, preconditions, postconditions, and exceptions raised.
  - Avoid self-aggrandizing adjectives like "robust," "advanced," "proper," or "sophisticated." Let the code speak for itself without unnecessary qualifiers.
  - Use progressive tense for comments (e.g., "Creating...", "Initializing...", "Checking...", "Using...") rather than imperative tense (e.g., "Create", "Initialize", "Check", "Use") or direct statements (e.g., "This function creates"). The progressive tense reads more naturally as though a human is describing an ongoing process.
  - Include detailed, accurate docstrings with the following characteristics:
    - Start with a one-line summary that clearly states the purpose
    - Follow with a detailed description explaining behavior and context
    - Document all parameters with accurate types and descriptions
    - Document return values and exceptions raised
    - Include clear usage examples with expected outputs
    - Use consistent formatting with proper indentation

  **Docstring Example**:
  ```python
  def vmap(fn: Callable[..., T], *, in_axes: Optional[Union[int, Dict[str, int]]] = 0) -> Callable[..., List[T]]:
      """Vectorizing a function across its inputs.
      
      Transforming a function that operates on single elements into one 
      that efficiently processes multiple inputs in parallel. The transformation preserves 
      the original function's semantics while enabling batch processing capabilities.
      
      Args:
          fn: The function to vectorize. Should accept and return dictionaries.
          in_axes: Specification of which inputs are batched and on which axis.
              If an integer, applies to all inputs. If a dict, specifies axes
              for specific keys. Keys not specified are treated as non-batch inputs.
      
      Returns:
          A vectorized version of the input function that handles batched inputs
          and produces batched outputs.
      
      Raises:
          ValueError: If inconsistent batch sizes are detected across inputs.
      
      Example:
          ```python
          def process_item(item: dict) -> dict:
              return {"result": item["value"] * 2}
          
          # Creating vectorized version
          batch_process = vmap(process_item)
          
          # Processing multiple items at once
          results = batch_process({"value": [1, 2, 3]})
          # results == {"result": [2, 4, 6]}
          ```
      """
  ```
- **Edits, Debugging, and Refactoring**:
  - Be incisive, clean, and minimal with all changes.
  - Make surgical, targeted changes that address the root issue.
  - Refactor with purpose; eliminate complexity, don't add it.
  - Always leave code cleaner than you found it.
  - Fix the underlying problem, not just the symptoms.
- **Design Principles**:
  - **Single Responsibility**: Each class and function should have exactly one reason to change.
  - **Open-Closed**: Code should be open for extension but closed for modification.
  - **Liskov Substitution**: Derived classes must be substitutable for their base classes.
  - **Interface Segregation**: Clients should not depend on interfaces they don't use.
  - **Dependency Inversion**: Depend on abstractions, not concretions.
  - Prioritize modularity, minimalism, and extensibility. Write streamlined, elegant code that is easy to reason about, test, and extend.
  - Centralize common functionality in well-designed core modules to reduce duplication and fragmentation.
  - Use dependency injection to make components testable and loosely coupled.
  - Decisively refactor and improve code quality without hesitation; avoid unnecessary backward compatibility constraints.
  - Favor named arguments for clarity and readability, especially for boolean parameters and functions with multiple arguments of the same type.
  - Prefer composition over inheritance when appropriate.
  - Design APIs that are hard to misuse and easy to use correctly.
  - Continuously strive for simplicity and elegance in design and implementation.

## Ember-Specific API Usage Guidelines

### Operator Patterns

1. **Standard Operator Definition Pattern**:
   ```python
   class MyOperator(Operator[MyInput, MyOutput]):
       # Class-level specification (required)
       # Use ClassVar when following the core API definition or for static analysis
       specification: ClassVar[Specification] = MySpecification()
       # Alternative syntax without ClassVar for simpler code:
       # specification = MySpecification()
       
       # Field declarations
       param1: str
       param2: int
       
       def __init__(self, *, param1: str, param2: int = 10) -> None:
           # Initialize fields
           self.param1 = param1
           self.param2 = param2
           
       def forward(self, *, inputs: MyInput) -> MyOutput:
           # Implementation
           return MyOutput(...)
   ```

   **Note on ClassVar usage:** ClassVar indicates the attribute belongs to the class, not instances. The core Operator definition uses ClassVar for specification, so it's most accurate to include it. However, in many cases, the code will work correctly without it since specification is accessed as a class attribute. When in doubt, include ClassVar for better static analysis.

2. **Container Operator with Sub-operators Pattern**:
   ```python
   @jit
   class PipelineOperator(Operator[Input, Output]):
       # Class-level specification with ClassVar for consistency
       specification: ClassVar[Specification] = PipelineSpecification()
       
       # Field declarations for sub-operators
       op1: FirstOperator
       op2: SecondOperator
       
       def __init__(self, *, config_param: str) -> None:
           # Create sub-operators
           self.op1 = FirstOperator(param=config_param)
           self.op2 = SecondOperator(param=config_param)
           
       def forward(self, *, inputs: Input) -> Output:
           # Connect operators in the pipeline
           intermediate = self.op1(inputs=inputs)
           return self.op2(inputs=intermediate)
   ```

3. **Operator Invocation Pattern**:
   - Preferred: Named argument style with explicit `inputs` parameter
     ```python
     result = my_operator(inputs=my_input)
     ```
   - Alternative: Dict-style input (when appropriate)
     ```python
     result = my_operator(inputs={"key": "value"})
     ```

4. **Operator Composition Patterns**:
   - Preferred: Container class with sub-operators (best for JIT optimization)
     ```python
     @jit
     class PipelineOperator(Operator[Input, Output]):
         # Implementation as above
     ```
   - Alternative: Sequential chaining (for simpler workflows)
     ```python
     def pipeline(inputs: Dict[str, Any]) -> Any:
         result1 = op1(inputs=inputs)
         result2 = op2(inputs=result1)
         return result2
     ```

### Model Registry Patterns

1. **Preferred Model Invocation Style**:
   ```python
   # Direct model invocation with namespace
   response = models.openai.gpt4o("What is the capital of France?")
   
   # With configuration builder
   model = (
       ModelBuilder()
       .temperature(0.7)
       .max_tokens(100)
       .build("anthropic:claude-3.5-sonnet")
   )
   response = model.generate(prompt="Your prompt here")
   ```

2. **Type-Safe Model References**:
   ```python
   # Use ModelEnum instead of string literals
   from ember.api.models import ModelEnum
   model = ModelAPI.from_enum(ModelEnum.OPENAI_GPT4O)
   response = model.generate(prompt="Your prompt here")
   ```

### JIT Optimization Patterns

1. **Standard JIT Usage Pattern**:
   ```python
   @jit
   class OptimizedOperator(Operator[Input, Output]):
       # Implementation...
       pass
   ```

2. **JIT with Specific Sample Input**:
   ```python
   @jit(sample_input={"key": "value"})
   class MyOperator(Operator[Input, Output]):
       # Implementation...
       pass
   ```

3. **Execution Context Control**:
   ```python
   # Configure execution options
   with execution_options(scheduler="parallel"):
       result = my_operator(inputs=my_input)
   ```

## Testing Guidelines

- **Avoid overmocking**: Test actual implementations rather than creating duplicate mock versions of core system components. Only mock external dependencies and boundaries.
- **Test real behavior**: Tests should validate how your code actually works, not how you think it works. Avoid parallel implementations in test code.
- **Root cause debugging**: Fix issues at the source rather than working around them in tests. When tests fail, understand and address the underlying problem, not just the symptom. Avoid brittle fixes like hasattr() checks or type workarounds that don't address the fundamental issue. Proper fixes typically involve updating the core implementation to handle edge cases correctly.
- **Integration testing**: Include tests that verify how components work together in the real system, not just in isolation.
- **Minimal test doubles**: Create the minimal test double needed for the test, preserving real behavior whenever possible.
- **Targeted mocking**: Mock at the level of the dependency being replaced, not at the level of the entire subsystem.
- **Test for correctness**: Focus tests on behavior correctness rather than implementation details that might change.
- **Language equivalence**: When implementing functionality in multiple languages (Python/Rust), verify behavior equivalence across identical inputs.

## Rust Implementation Guidelines

Rust code must adhere to the same engineering excellence standards as Python code, with additional considerations for performance optimization, memory safety, and FFI boundaries.

### Core Principles

- **Performance with correctness**: Optimize critical paths without compromising correctness or clarity.
- **Incisive minimalism**: Write clean, minimal code with clear intent and no unnecessary abstraction.
- **API consistency**: Maintain exact semantic equivalence with Python interfaces when implementing Rust versions.
- **Type richness**: Use Rust's type system to encode invariants and prevent errors at compile time.
- **Fearless concurrency**: Leverage Rust's ownership model to enable safe, efficient multithreaded code.

### Code Style

- **Naming and organization**: Follow Rust convention with `snake_case` for functions/variables and `CamelCase` for types/traits. Structure modules with clear responsibility boundaries.
- **Documentation**: Every public API must have comprehensive rustdoc comments equivalent to Python docstrings - purpose, behavior, parameters, returns, errors, and examples.
- **Error handling**: Use domain-specific error types that convert appropriately to Python exceptions at language boundaries.
- **Memory management**: Design for efficiency by minimizing allocations, using stack when possible, reusing buffers, and optimizing for cache locality.

### Python Integration

- **Interface consistency**: Python-facing APIs must match the original Python interfaces exactly, including named parameters and error behaviors.
- **Type conversion**: Implement seamless bidirectional conversion between Rust and Python types, handling complex nested structures correctly.
- **GIL awareness**: Properly manage Python GIL requirements while maximizing opportunities for parallelism.

### Testing for Rust Components

- **Property-based testing**: Verify implementation correctness across wide input ranges using fuzzing and property-based testing.
- **Parallel validation**: Run identical inputs through both implementations to verify equivalent outputs.
- **Performance verification**: Create benchmarks that quantify the expected performance improvements.

Use the same SOLID principles, clarity standards, and engineering discipline required for Python code, while thoughtfully exploiting Rust's unique strengths.