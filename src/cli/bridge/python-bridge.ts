/**
 * Python Bridge Module
 * 
 * This module serves as the interface between the Node.js CLI and the Python Ember backend.
 * It follows the Adapter pattern to provide a clean interface.
 */

import { PythonBridge } from 'python-bridge';
import { isDebugMode } from '../utils/options';
import chalk from 'chalk';
import path from 'path';
import fs from 'fs';
import { 
  PythonBridgeError, 
  createPythonBridgeError, 
  EmberCliError,
  PythonBridgeErrorCodes,
  handleError 
} from '../errors';

// Define types for our Python interface
export interface EmberPythonBridge {
  /**
   * Initialize the bridge
   */
  initialize(): Promise<void>;
  
  /**
   * Get the Ember version
   */
  getVersion(): Promise<string>;
  
  /**
   * List all providers
   */
  listProviders(): Promise<string[]>;
  
  /**
   * List all models, optionally filtered by provider
   */
  listModels(provider?: string): Promise<string[]>;
  
  /**
   * Get detailed information about a model
   */
  getModelInfo(modelId: string): Promise<any>;
  
  /**
   * Get information about a provider
   */
  getProviderInfo(providerId: string): Promise<any>;
  
  /**
   * Invoke a model with a prompt
   */
  invokeModel(modelId: string, prompt: string, options?: Record<string, any>): Promise<any>;
  
  /**
   * Create a new project
   */
  createProject(projectName: string, options: Record<string, any>): Promise<void>;
  
  /**
   * Get usage statistics
   */
  getUsageStats(): Promise<any>;
  
  /**
   * Clean up resources
   */
  cleanup(): Promise<void>;
}

/**
 * Error coming from Python execution
 */
interface PythonErrorResult {
  error: string;
  error_type?: string;
  traceback?: string[];
}

/**
 * Result wrapper to handle errors consistently
 */
interface PythonResult<T> {
  success: boolean;
  data?: T;
  error?: PythonErrorResult;
}

// Private implementation of the bridge
class PythonBridgeImpl implements EmberPythonBridge {
  private bridge: PythonBridge;
  private initialized: boolean = false;
  
  constructor() {
    try {
      this.bridge = new PythonBridge({
        python: 'python',
        env: process.env
      });
      
      // Setup error handling
      this.bridge.stderr.on('data', (data: Buffer) => {
        const errorMsg = data.toString();
        if (isDebugMode()) {
          console.error(chalk.red('Python Error:'), errorMsg);
        }
      });
    } catch (error: any) {
      throw createPythonBridgeError(
        `Failed to create Python bridge: ${error.message}`,
        PythonBridgeErrorCodes.PYTHON_NOT_FOUND,
        { cause: error }
      );
    }
  }
  
  async initialize(): Promise<void> {
    if (this.initialized) return;
    
    try {
      // Execute Python setup code with enhanced error handling
      await this.evaluatePython(`
import sys
import os
import json
import traceback

# Create a standard error response format
def format_error(e):
    error_type = type(e).__name__
    error_msg = str(e)
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    return json.dumps({
        "success": False,
        "error": {
            "error_type": error_type,
            "error": error_msg,
            "traceback": tb
        }
    })

# Function to wrap results in a standard format
def format_result(result):
    return json.dumps({
        "success": True,
        "data": result
    })

try:
    import ember
except ImportError as e:
    print(format_error(e))
    raise e

# Add ember modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize the service globally
service = None

def init_service(tracking=False):
    global service
    try:
        service = ember.init(usage_tracking=tracking)
        return format_result(True)
    except Exception as e:
        return format_error(e)
`);
      
      // Initialize ember service
      const initResult = await this.evaluatePython<boolean>('init_service(True)');
      
      this.initialized = true;
    } catch (error) {
      if (error instanceof EmberCliError) {
        throw error;
      }
      
      throw createPythonBridgeError(
        `Failed to initialize Python bridge: ${error}`,
        PythonBridgeErrorCodes.BRIDGE_INITIALIZATION_FAILED, 
        { cause: error as Error }
      );
    }
  }
  
  async getVersion(): Promise<string> {
    await this.ensureInitialized();
    return await this.evaluatePython<string>('format_result(ember.__version__)');
  }
  
  async listProviders(): Promise<string[]> {
    await this.ensureInitialized();
    return await this.evaluatePython<string[]>(`
try:
    providers = service.registry.list_providers()
    print(format_result(providers))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async listModels(provider?: string): Promise<string[]> {
    await this.ensureInitialized();
    
    if (provider) {
      return await this.evaluatePython<string[]>(`
try:
    models = service.registry.list_models()
    filtered_models = [m for m in models if m.startswith(f"${provider}:")]
    print(format_result(filtered_models))
except Exception as e:
    print(format_error(e))
    raise e
`);
    } else {
      return await this.evaluatePython<string[]>(`
try:
    models = service.registry.list_models()
    print(format_result(models))
except Exception as e:
    print(format_error(e))
    raise e
`);
    }
  }
  
  async getModelInfo(modelId: string): Promise<any> {
    await this.ensureInitialized();
    
    return await this.evaluatePython<any>(`
try:
    model_info = service.registry.get_model_info(${JSON.stringify(modelId)})
    print(format_result(model_info.model_dump()))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async getProviderInfo(providerId: string): Promise<any> {
    await this.ensureInitialized();
    
    return await this.evaluatePython<any>(`
try:
    provider_info = service.registry.get_provider_info(${JSON.stringify(providerId)})
    print(format_result(provider_info.model_dump()))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async invokeModel(modelId: string, prompt: string, options: Record<string, any> = {}): Promise<any> {
    await this.ensureInitialized();
    
    const optionsJson = JSON.stringify(options);
    
    return await this.evaluatePython<any>(`
try:
    options = json.loads(${JSON.stringify(optionsJson)})
    response = service(${JSON.stringify(modelId)}, ${JSON.stringify(prompt)})
    usage = service.usage_service.get_last_usage()
    
    result = {
        "data": response.data,
        "usage": None if usage is None else {
            "tokens": usage.tokens,
            "cost": usage.cost
        }
    }
    print(format_result(result))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async createProject(projectName: string, options: Record<string, any>): Promise<void> {
    await this.ensureInitialized();
    
    const optionsJson = JSON.stringify(options);
    
    await this.evaluatePython<void>(`
try:
    import os
    
    options = json.loads(${JSON.stringify(optionsJson)})
    dir_name = ${JSON.stringify(projectName)}
    
    # Check if dir exists
    if os.path.exists(dir_name):
        raise Exception(f"Directory '{dir_name}' already exists")
    
    # Create project directory
    os.makedirs(dir_name)
    
    # Create simple example file
    example_file = os.path.join(dir_name, "ember_example.py")
    with open(example_file, "w") as f:
        f.write("""#!/usr/bin/env python
'''
Ember Quickstart Example
'''

import ember
from typing import ClassVar

from ember.core.registry.operator.base import Operator
from ember.core.registry.prompt_specification import Specification
from ember.core.types.ember_model import EmberModel
from ember.core import non


# Define structured input/output types
class QueryInput(EmberModel):
    query: str
    
class QueryOutput(EmberModel):
    answer: str
    

# Define the specification
class SimpleQASpecification(Specification):
    input_model = QueryInput
    output_model = QueryOutput
    prompt_template = "Please answer this question: {query}"


# Create a simple operator
class SimpleQA(Operator[QueryInput, QueryOutput]):
    # Class-level specification declaration
    specification: ClassVar[Specification] = SimpleQASpecification()
    
    # Class-level field declarations
    lm: non.UniformEnsemble
    
    def __init__(self, model_name: str = "openai:gpt-4o-mini"):
        # Initialize fields
        self.lm = non.UniformEnsemble(
            num_units=1,
            model_name=model_name
        )
    
    def forward(self, *, inputs: QueryInput) -> QueryOutput:
        # Process the query through the LLM
        response = self.lm(inputs={"query": inputs.query})
        
        # Return structured output
        return QueryOutput(answer=response["responses"][0])


def main():
    '''Main entry point'''
    # Initialize Ember with one line
    service = ember.init()
    
    # Create and use our QA system
    qa = SimpleQA()
    result = qa(inputs={"query": "What is the capital of France?"})
    
    print(f"Answer: {result.answer}")
    
    # Direct model invocation is also possible
    direct_response = service("openai:gpt-4o-mini", "What is 2+2?")
    print(f"Direct response: {direct_response.data}")


if __name__ == "__main__":
    main()
""")
    
    # Create README.md
    readme_file = os.path.join(dir_name, "README.md")
    with open(readme_file, "w") as f:
        f.write(f"""# {dir_name}
    
A project created with Ember CLI.
    
## Setup
    
1. Install dependencies:
```bash
pip install "ember-ai[openai]"
```
    
2. Set up your API keys:
```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
    
# For Windows PowerShell
$env:OPENAI_API_KEY="your-openai-key"
```
    
## Run the example
    
```bash
python ember_example.py
```
""")
    
    # Create .env.example file
    env_file = os.path.join(dir_name, ".env.example")
    with open(env_file, "w") as f:
        f.write("""# API Keys - Replace with your actual keys and rename to .env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
""")
    
    print(format_result(None))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async getUsageStats(): Promise<any> {
    await this.ensureInitialized();
    
    return await this.evaluatePython<any>(`
try:
    usage_stats = service.usage_service.get_usage_stats()
    print(format_result(usage_stats))
except Exception as e:
    print(format_error(e))
    raise e
`);
  }
  
  async cleanup(): Promise<void> {
    if (this.initialized) {
      try {
        this.bridge.end();
      } catch (error) {
        // Silently handle cleanup errors
        if (isDebugMode()) {
          console.error('Error during Python bridge cleanup:', error);
        }
      } finally {
        this.initialized = false;
      }
    }
  }
  
  // Utility methods
  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
  }
  
  /**
   * Evaluate Python code and process the result
   * 
   * @param code Python code to evaluate
   * @returns Processed result with error handling
   */
  private async evaluatePython<T>(code: string): Promise<T> {
    try {
      const resultStr = await this.bridge.eval(code);
      
      try {
        // Parse the result
        const result = JSON.parse(resultStr) as PythonResult<T>;
        
        if (!result.success && result.error) {
          // Handle Python exception
          throw EmberCliError.fromPythonError(
            result.error.error_type ? `${result.error.error_type}: ${result.error.error}` : result.error.error
          );
        }
        
        return result.data as T;
      } catch (parseError) {
        // If the result is not valid JSON, it might be a direct string value
        if (parseError instanceof SyntaxError) {
          return resultStr as unknown as T;
        }
        throw parseError;
      }
    } catch (error) {
      // If error is already an EmberCliError, just rethrow it
      if (error instanceof EmberCliError) {
        throw error;
      }
      
      // Otherwise, create a new PythonBridgeError
      throw createPythonBridgeError(
        `Python bridge error: ${error}`,
        PythonBridgeErrorCodes.PYTHON_EXECUTION_ERROR,
        { cause: error as Error }
      );
    }
  }
}

// Singleton instance of the bridge
let bridgeInstance: EmberPythonBridge | null = null;

/**
 * Get the Python bridge instance
 * Uses the Singleton pattern to ensure we only have one bridge
 */
export function getPythonBridge(): EmberPythonBridge {
  if (!bridgeInstance) {
    bridgeInstance = new PythonBridgeImpl();
  }
  
  return bridgeInstance;
}

/**
 * Clean up bridge resources before exit
 */
export async function cleanupPythonBridge(): Promise<void> {
  if (bridgeInstance) {
    await bridgeInstance.cleanup();
    bridgeInstance = null;
  }
}

// Clean up on process exit
process.on('exit', () => {
  if (bridgeInstance) {
    // We can't use async here, so we do a sync cleanup
    // This is not ideal but necessary for process.exit
    try {
      (bridgeInstance as any).bridge?.end();
    } catch (error) {
      // Silently ignore cleanup errors on exit
    }
  }
});