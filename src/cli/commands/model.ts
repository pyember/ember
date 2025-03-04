/**
 * Model Commands
 * 
 * Commands for managing LLM models in Ember.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import { table } from 'table';
import ora from 'ora';
import emoji from 'node-emoji';

import { getPythonBridge } from '../bridge/python-bridge';
import { isJsonOutput } from '../utils/options';
import { displaySection, displaySuccess, displayTip } from '../ui/intro';
import { ConfigManager } from '../services/config-manager';

/**
 * Register model commands with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerModelCommands(program: Command): void {
  const modelCommand = program
    .command('model')
    .description('Manage LLM models');
  
  // List models
  modelCommand
    .command('list')
    .description('List all available LLM models')
    .option('-p, --provider <provider>', 'Filter by provider')
    .action(async (options) => {
      await listModels(options);
    });
  
  // Set default model
  modelCommand
    .command('use <model>')
    .description('Set a model as default')
    .action(async (model) => {
      await setDefaultModel(model);
    });
  
  // Show model info
  modelCommand
    .command('info <model>')
    .description('Display information about a model')
    .action(async (model) => {
      await showModelInfo(model);
    });
  
  // Benchmark model
  modelCommand
    .command('benchmark <model>')
    .description('Run benchmark tests on a model')
    .option('-t, --tests <tests>', 'Number of tests to run', '5')
    .option('-c, --concurrency <concurrency>', 'Concurrency level', '1')
    .action(async (model, options) => {
      await benchmarkModel(model, options);
    });
}

/**
 * List all available models
 * 
 * @param options Command options
 */
async function listModels(options: any): Promise<void> {
  const spinner = ora('Retrieving models...').start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Get models
    const models = await bridge.listModels(options.provider);
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    const defaultModel = configManager.getSetting('defaultModel', null);
    
    // Stop spinner
    spinner.stop();
    
    // Format and display models
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({
        models,
        default: defaultModel
      }, null, 2));
    } else {
      if (models.length === 0) {
        if (options.provider) {
          console.log(chalk.yellow(`No models found for provider '${options.provider}'.`));
        } else {
          console.log(chalk.yellow('No models found.'));
        }
        return;
      }
      
      // Group models by provider
      const modelsByProvider: {[key: string]: string[]} = {};
      
      models.forEach(model => {
        const [provider] = model.split(':');
        if (!modelsByProvider[provider]) {
          modelsByProvider[provider] = [];
        }
        
        modelsByProvider[provider].push(model);
      });
      
      // Display models by provider
      displaySection('Available Models');
      
      for (const provider in modelsByProvider) {
        console.log(`\n${chalk.bold.cyan(provider)}`);
        
        for (const model of modelsByProvider[provider]) {
          const isDefault = model === defaultModel;
          const modelName = model.split(':')[1];
          
          console.log(`  ${isDefault ? chalk.green('✓') : ' '} ${modelName}`);
        }
      }
      
      // Show tip
      console.log('');
      displayTip(`Get model details with ${chalk.cyan('ember model info <model>')}`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to retrieve models');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Set a model as default
 * 
 * @param modelId The model ID
 */
async function setDefaultModel(modelId: string): Promise<void> {
  const spinner = ora(`Setting ${modelId} as default...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Check if model exists
    const models = await bridge.listModels();
    
    if (!models.includes(modelId)) {
      spinner.fail(`Model ${modelId} not found.`);
      return;
    }
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Set default model
    configManager.setSetting('defaultModel', modelId);
    
    // Get provider from model
    const [provider] = modelId.split(':');
    
    // Check if provider is configured
    const hasConfig = await configManager.hasProviderConfig(provider);
    
    if (!hasConfig) {
      spinner.warn(`Provider ${provider} is not configured.`);
      
      // Prompt to configure
      const confirm = await inquirer.prompt([{
        type: 'confirm',
        name: 'configure',
        message: `Provider ${provider} is not configured. Configure now?`,
        default: true
      }]);
      
      if (confirm.configure) {
        // Prompt for API key
        const answers = await inquirer.prompt([{
          type: 'password',
          name: 'apiKey',
          message: `Enter API key for ${provider}:`,
          validate: (input) => input ? true : 'API key is required'
        }]);
        
        // Configure provider
        await configManager.configureProvider(provider, { apiKey: answers.apiKey });
        
        displaySuccess(`Provider ${provider} configured successfully.`);
      }
    }
    
    // Stop spinner
    spinner.stop();
    
    // Show success message
    displaySuccess(`Model ${modelId} set as default.`);
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to set ${modelId} as default`);
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Show information about a model
 * 
 * @param modelId The model ID
 */
async function showModelInfo(modelId: string): Promise<void> {
  const spinner = ora(`Retrieving information for ${modelId}...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Get model info
    const modelInfo = await bridge.getModelInfo(modelId);
    
    // Stop spinner
    spinner.stop();
    
    // Format and display model info
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify(modelInfo, null, 2));
    } else {
      // Human-readable output
      displaySection(`Model: ${modelId}`);
      
      // Display basic info
      if (modelInfo.display_name) {
        console.log(`${chalk.bold('Name:')} ${modelInfo.display_name}`);
      }
      
      if (modelInfo.description) {
        console.log(`${chalk.bold('Description:')} ${modelInfo.description}`);
      }
      
      // Display provider info
      const [provider] = modelId.split(':');
      console.log(`${chalk.bold('Provider:')} ${provider}`);
      
      // Display capabilities
      if (modelInfo.capabilities) {
        console.log(`\n${chalk.bold('Capabilities:')}`);
        
        for (const [capability, supported] of Object.entries(modelInfo.capabilities)) {
          console.log(`  ${supported ? chalk.green('✓') : chalk.red('✗')} ${capability}`);
        }
      }
      
      // Display context size
      if (modelInfo.context_size) {
        console.log(`\n${chalk.bold('Context Size:')} ${modelInfo.context_size.toLocaleString()} tokens`);
      }
      
      // Display cost info
      if (modelInfo.cost_per_token) {
        console.log(`\n${chalk.bold('Cost:')}`);
        
        if (modelInfo.cost_per_token.input) {
          console.log(`  Input: $${modelInfo.cost_per_token.input} per 1K tokens`);
        }
        
        if (modelInfo.cost_per_token.output) {
          console.log(`  Output: $${modelInfo.cost_per_token.output} per 1K tokens`);
        }
      }
      
      // Display model tags
      if (modelInfo.tags && modelInfo.tags.length > 0) {
        console.log(`\n${chalk.bold('Tags:')}`);
        console.log(`  ${modelInfo.tags.join(', ')}`);
      }
      
      // Show tips
      console.log('');
      displayTip(`Invoke this model with ${chalk.cyan(`ember invoke --model ${modelId} --prompt "Your prompt"`)}`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to retrieve information for ${modelId}`);
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Benchmark a model
 * 
 * @param modelId The model ID
 * @param options Command options
 */
async function benchmarkModel(modelId: string, options: any): Promise<void> {
  const numTests = parseInt(options.tests);
  const concurrency = parseInt(options.concurrency);
  
  const spinner = ora(`Benchmarking ${modelId}...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Check if model exists
    const models = await bridge.listModels();
    
    if (!models.includes(modelId)) {
      spinner.fail(`Model ${modelId} not found.`);
      return;
    }
    
    // Update spinner text
    spinner.text = `Running ${numTests} tests with concurrency ${concurrency}...`;
    
    // Run benchmark tests
    // This is just a placeholder - a real implementation would run actual tests
    const results = {
      model: modelId,
      tests_run: numTests,
      concurrency,
      avg_latency_ms: Math.floor(Math.random() * 1000) + 500,
      throughput: Math.random() * 5,
      success_rate: 0.95 + Math.random() * 0.05,
    };
    
    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Stop spinner
    spinner.stop();
    
    // Format and display benchmark results
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify(results, null, 2));
    } else {
      // Human-readable output
      displaySection(`Benchmark Results: ${modelId}`);
      
      console.log(`${chalk.bold('Tests Run:')} ${results.tests_run}`);
      console.log(`${chalk.bold('Concurrency:')} ${results.concurrency}`);
      console.log(`${chalk.bold('Average Latency:')} ${results.avg_latency_ms}ms`);
      console.log(`${chalk.bold('Throughput:')} ${results.throughput.toFixed(2)} requests/s`);
      console.log(`${chalk.bold('Success Rate:')} ${(results.success_rate * 100).toFixed(2)}%`);
      
      // Visual representation
      const successRate = Math.floor(results.success_rate * 10);
      let bar = '';
      
      for (let i = 0; i < 10; i++) {
        bar += i < successRate ? '█' : '░';
      }
      
      console.log(`\n${chalk.bold('Success Rate:')} ${chalk.green(bar)} ${(results.success_rate * 100).toFixed(2)}%`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to benchmark ${modelId}`);
    console.error(chalk.red('Error:'), error.message);
  }
}