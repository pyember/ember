/**
 * Invoke Command
 * 
 * Command for invoking LLM models directly from the CLI.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import ora from 'ora';
import emoji from 'node-emoji';
import fs from 'fs';
import path from 'path';

import { getPythonBridge } from '../bridge/python-bridge';
import { isJsonOutput, isDebugMode } from '../utils/options';
import { displaySection, displaySuccess, displayTip } from '../ui/intro';
import { ConfigManager } from '../services/config-manager';
import { 
  handleError, 
  createModelError, 
  createValidationError, 
  ModelErrorCodes, 
  GeneralErrorCodes,
  tryCatch
} from '../errors';

/**
 * Register the invoke command with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerInvokeCommand(program: Command): void {
  program
    .command('invoke')
    .description('Invoke a model with a prompt')
    .option('-m, --model <model>', 'Model ID to use')
    .option('-p, --prompt <prompt>', 'Prompt text to send to the model')
    .option('-f, --file <file>', 'Read prompt from file')
    .option('-s, --system <s>', 'System prompt (for chat models)')
    .option('-t, --temperature <temp>', 'Temperature setting (0.0-2.0)', '1.0')
    .option('-u, --show-usage', 'Show token usage statistics')
    .option('-o, --output <file>', 'Save output to file')
    .option('-r, --raw', 'Display raw output without formatting')
    .option('--stream', 'Stream the response token by token')
    .action(async (options) => {
      try {
        await invokeModel(options);
      } catch (error) {
        handleError(error, {}, true);
      }
    });
}

/**
 * Invoke a model with a prompt
 * 
 * @param options Command options
 */
async function invokeModel(options: any): Promise<void> {
  // Get config manager
  const configManager = ConfigManager.getInstance();
  
  // Get model ID
  let modelId = options.model;
  
  if (!modelId) {
    // Try to get default model
    modelId = configManager.getSetting('defaultModel', null);
    
    if (!modelId) {
      // If no model specified and no default, prompt for model
      const bridge = getPythonBridge();
      await bridge.initialize();
      
      const models = await bridge.listModels();
      
      if (models.length === 0) {
        throw createModelError(
          'No models available. Make sure your providers are configured correctly.',
          ModelErrorCodes.MODEL_NOT_FOUND,
          {
            suggestions: [
              'Configure your provider API keys with `ember config set provider.apiKey YOUR_KEY`',
              'Check that at least one language model provider is installed'
            ]
          }
        );
      }
      
      const answer = await inquirer.prompt([{
        type: 'list',
        name: 'model',
        message: 'Select a model:',
        choices: models
      }]);
      
      modelId = answer.model;
    }
  }
  
  // Get prompt
  let prompt = options.prompt;
  
  if (options.file) {
    // Read prompt from file
    const filePath = path.resolve(options.file);
    
    if (!fs.existsSync(filePath)) {
      throw createValidationError(
        `File not found: ${filePath}`,
        GeneralErrorCodes.FILE_NOT_FOUND,
        {
          suggestions: [
            'Check that the file exists and the path is correct',
            'Use an absolute path or a path relative to the current directory'
          ]
        }
      );
    }
    
    try {
      prompt = fs.readFileSync(filePath, 'utf8');
    } catch (error) {
      throw createValidationError(
        `Failed to read file: ${filePath}`,
        GeneralErrorCodes.FILE_NOT_FOUND,
        { cause: error as Error }
      );
    }
  }
  
  if (!prompt) {
    // If no prompt, prompt for it
    const answer = await inquirer.prompt([{
      type: 'editor',
      name: 'prompt',
      message: 'Enter your prompt:',
      validate: (input) => input ? true : 'Prompt is required'
    }]);
    
    prompt = answer.prompt;
  }
  
  // Validate prompt
  if (!prompt || prompt.trim() === '') {
    throw createValidationError('Prompt cannot be empty', GeneralErrorCodes.INVALID_ARGUMENT);
  }
  
  // Get system prompt if provided
  const systemPrompt = options.system;
  
  // Get temperature
  let temperature: number;
  try {
    temperature = parseFloat(options.temperature);
    
    // Validate temperature
    if (isNaN(temperature) || temperature < 0 || temperature > 2) {
      throw createValidationError(
        'Temperature must be a number between 0.0 and 2.0',
        GeneralErrorCodes.INVALID_ARGUMENT
      );
    }
  } catch (error) {
    if (error instanceof Error && !(error.message.includes('Temperature must be'))) {
      throw createValidationError(
        `Invalid temperature value: ${options.temperature}`,
        GeneralErrorCodes.INVALID_ARGUMENT
      );
    }
    throw error;
  }
  
  // Prepare invoke options
  const invokeOptions: Record<string, any> = {
    temperature
  };
  
  if (systemPrompt) {
    invokeOptions.system = systemPrompt;
  }
  
  // Show spinner if not streaming
  const spinner = options.stream ? null : ora(`Invoking ${modelId}...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Invoke model
    const response = await bridge.invokeModel(modelId, prompt, invokeOptions);
    
    // Stop spinner
    if (spinner) {
      spinner.stop();
    }
    
    // Save output to file if requested
    if (options.output) {
      const outputPath = path.resolve(options.output);
      try {
        fs.writeFileSync(outputPath, response.data);
        displaySuccess(`Output saved to ${outputPath}`);
      } catch (error) {
        throw createValidationError(
          `Failed to write to file: ${outputPath}`,
          GeneralErrorCodes.PERMISSION_DENIED,
          { cause: error as Error }
        );
      }
    }
    
    // Format and display response
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify(response, null, 2));
    } else {
      // Display delineation between prompt and response for clarity
      if (!options.raw) {
        const divider = '─'.repeat(process.stdout.columns || 80);
        console.log(chalk.dim(divider));
      }
      
      // Display response
      console.log(options.raw ? response.data : chalk.white(response.data));
      
      // Show usage info if requested
      if (options.showUsage && response.usage) {
        console.log('');
        displaySection('Usage Information');
        console.log(`${chalk.bold('Tokens:')} ${response.usage.tokens}`);
        console.log(`${chalk.bold('Cost:')} $${response.usage.cost.toFixed(6)}`);
      }
    }
  } catch (error) {
    // Stop spinner if still running
    if (spinner) {
      spinner.stop();
    }
    
    // Rethrow the error to be handled by the command handler
    throw error;
  }
}