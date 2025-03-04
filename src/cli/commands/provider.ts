/**
 * Provider Commands
 * 
 * Manage and configure providers for Ember.
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
 * Register provider commands with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerProviderCommands(program: Command): void {
  const providerCommand = program
    .command('provider')
    .description('Manage LLM providers');
  
  // List providers
  providerCommand
    .command('list')
    .description('List all available LLM providers')
    .action(async () => {
      await listProviders();
    });
  
  // Configure provider
  providerCommand
    .command('configure <provider>')
    .description('Configure a provider with API keys')
    .option('-k, --key <key>', 'API key (or omit to be prompted)')
    .option('-f, --force', 'Overwrite existing configuration')
    .action(async (provider, options) => {
      await configureProvider(provider, options);
    });
  
  // Show provider info
  providerCommand
    .command('info <provider>')
    .description('Display information about a provider')
    .action(async (provider) => {
      await showProviderInfo(provider);
    });
  
  // Set default provider
  providerCommand
    .command('use <provider>')
    .description('Set a provider as default')
    .action(async (provider) => {
      await setDefaultProvider(provider);
    });
}

/**
 * List all available providers
 */
async function listProviders(): Promise<void> {
  const spinner = ora('Retrieving providers...').start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Get providers
    const providers = await bridge.listProviders();
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    const defaultProvider = await configManager.getDefaultProvider();
    
    // Stop spinner
    spinner.stop();
    
    // Format and display providers
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({
        providers,
        default: defaultProvider
      }, null, 2));
    } else {
      if (providers.length === 0) {
        console.log(chalk.yellow('No providers found.'));
        return;
      }
      
      // Create table data
      const tableData = [
        [chalk.bold('Provider'), chalk.bold('Status'), chalk.bold('Default')]
      ];
      
      // Add provider rows
      for (const provider of providers) {
        const isDefault = provider === defaultProvider;
        const hasConfig = await configManager.hasProviderConfig(provider);
        
        tableData.push([
          provider,
          hasConfig ? chalk.green('Configured') : chalk.yellow('Not Configured'),
          isDefault ? chalk.green('✓') : ''
        ]);
      }
      
      // Display table
      console.log(table(tableData, {
        border: {
          topBody: `─`,
          topJoin: `┬`,
          topLeft: `┌`,
          topRight: `┐`,
          bottomBody: `─`,
          bottomJoin: `┴`,
          bottomLeft: `└`,
          bottomRight: `┘`,
          bodyLeft: `│`,
          bodyRight: `│`,
          bodyJoin: `│`,
          joinBody: `─`,
          joinLeft: `├`,
          joinRight: `┤`,
          joinJoin: `┼`
        }
      }));
      
      // Show tip
      displayTip(`Configure a provider with ${chalk.cyan('ember provider configure <provider>')}`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to retrieve providers');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Configure a provider with API keys
 * 
 * @param provider The provider ID
 * @param options Command options
 */
async function configureProvider(provider: string, options: any): Promise<void> {
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Check if provider is already configured
    const hasConfig = await configManager.hasProviderConfig(provider);
    
    if (hasConfig && !options.force) {
      const confirm = await inquirer.prompt([{
        type: 'confirm',
        name: 'overwrite',
        message: `Provider ${provider} is already configured. Overwrite?`,
        default: false
      }]);
      
      if (!confirm.overwrite) {
        console.log(chalk.yellow('Configuration cancelled.'));
        return;
      }
    }
    
    // Get API key
    let apiKey = options.key;
    
    if (!apiKey) {
      // Prompt for API key
      const answers = await inquirer.prompt([{
        type: 'password',
        name: 'apiKey',
        message: `Enter API key for ${provider}:`,
        validate: (input) => input ? true : 'API key is required'
      }]);
      
      apiKey = answers.apiKey;
    }
    
    // Configure provider
    await configManager.configureProvider(provider, { apiKey });
    
    // Show success message
    displaySuccess(`Provider ${provider} configured successfully.`);
    
    // If no default provider is set, set this one
    const defaultProvider = await configManager.getDefaultProvider();
    if (!defaultProvider) {
      await configManager.setDefaultProvider(provider);
      displaySuccess(`Provider ${provider} set as default.`);
    }
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Show information about a provider
 * 
 * @param provider The provider ID
 */
async function showProviderInfo(provider: string): Promise<void> {
  const spinner = ora(`Retrieving information for ${provider}...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Get provider info
    const providerInfo = await bridge.getProviderInfo(provider);
    
    // Stop spinner
    spinner.stop();
    
    // Format and display provider info
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify(providerInfo, null, 2));
    } else {
      // Check for error
      if (providerInfo.error) {
        console.log(chalk.red('Error:'), providerInfo.error);
        return;
      }
      
      // Human-readable output
      displaySection(`Provider: ${provider}`);
      
      // Display basic info
      if (providerInfo.display_name) {
        console.log(`${chalk.bold('Name:')} ${providerInfo.display_name}`);
      }
      
      if (providerInfo.description) {
        console.log(`${chalk.bold('Description:')} ${providerInfo.description}`);
      }
      
      if (providerInfo.website) {
        console.log(`${chalk.bold('Website:')} ${providerInfo.website}`);
      }
      
      // Display models
      if (providerInfo.models && providerInfo.models.length > 0) {
        console.log(`\n${chalk.bold('Available Models:')}`);
        for (const model of providerInfo.models) {
          console.log(`  • ${model}`);
        }
      }
      
      // Display authentication info
      console.log(`\n${chalk.bold('Authentication:')}`);
      console.log(`  Environment Variable: ${chalk.cyan(provider.toUpperCase() + '_API_KEY')}`);
      
      // Get config manager
      const configManager = ConfigManager.getInstance();
      const hasConfig = await configManager.hasProviderConfig(provider);
      
      console.log(`  Status: ${hasConfig ? chalk.green('Configured') : chalk.yellow('Not Configured')}`);
      
      // Show tips
      console.log('');
      if (!hasConfig) {
        displayTip(`Configure this provider with ${chalk.cyan(`ember provider configure ${provider}`)}`);
      }
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to retrieve information for ${provider}`);
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Set a provider as default
 * 
 * @param provider The provider ID
 */
async function setDefaultProvider(provider: string): Promise<void> {
  const spinner = ora(`Setting ${provider} as default...`).start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Check if provider exists
    const providers = await bridge.listProviders();
    
    if (!providers.includes(provider)) {
      spinner.fail(`Provider ${provider} not found.`);
      return;
    }
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
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
        await configureProvider(provider, {});
      } else {
        console.log(chalk.yellow('Default provider not set.'));
        return;
      }
    }
    
    // Set default provider
    await configManager.setDefaultProvider(provider);
    
    // Stop spinner
    spinner.stop();
    
    // Show success message
    displaySuccess(`Provider ${provider} set as default.`);
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to set ${provider} as default`);
    console.error(chalk.red('Error:'), error.message);
  }
}