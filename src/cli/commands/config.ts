/**
 * Configuration Commands
 * 
 * Commands for managing Ember CLI configuration settings.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import ora from 'ora';
import fs from 'fs';
import path from 'path';

import { isJsonOutput } from '../utils/options';
import { displaySection, displaySuccess, displayWarning, displayTip } from '../ui/intro';
import { ConfigManager } from '../services/config-manager';

/**
 * Register configuration commands with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerConfigCommands(program: Command): void {
  const configCommand = program
    .command('config')
    .description('Manage Ember CLI configuration');
  
  // List config
  configCommand
    .command('list')
    .description('List all configuration settings')
    .option('--show-keys', 'Show API keys (not recommended)')
    .action(async (options) => {
      await listConfig(options);
    });
  
  // Set config
  configCommand
    .command('set <key> <value>')
    .description('Set a configuration value')
    .action(async (key, value) => {
      await setConfig(key, value);
    });
  
  // Get config
  configCommand
    .command('get <key>')
    .description('Get a configuration value')
    .action(async (key) => {
      await getConfig(key);
    });
  
  // Reset config
  configCommand
    .command('reset')
    .description('Reset all configuration to defaults')
    .option('-f, --force', 'Skip confirmation')
    .action(async (options) => {
      await resetConfig(options);
    });
  
  // Import/export config
  configCommand
    .command('export <file>')
    .description('Export configuration to a file')
    .action(async (file) => {
      await exportConfig(file);
    });
  
  configCommand
    .command('import <file>')
    .description('Import configuration from a file')
    .option('-f, --force', 'Overwrite existing configuration')
    .action(async (file, options) => {
      await importConfig(file, options);
    });
  
  // Configure usage tracking
  configCommand
    .command('usage-tracking <enabled>')
    .description('Enable or disable usage tracking')
    .action(async (enabled) => {
      await configureUsageTracking(enabled === 'true' || enabled === 'on' || enabled === '1');
    });
}

/**
 * List all configuration settings
 * 
 * @param options Command options
 */
async function listConfig(options: any): Promise<void> {
  const spinner = ora('Retrieving configuration...').start();
  
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Get configuration
    const defaultProvider = configManager.getDefaultProvider();
    const defaultModel = configManager.getSetting('defaultModel', null);
    const usageTracking = configManager.isUsageTrackingEnabled();
    const providers = Object.keys(configManager.getSetting('providers', {}));
    
    // Stop spinner
    spinner.stop();
    
    // Format and display configuration
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({
        defaultProvider,
        defaultModel,
        usageTracking,
        providers,
        settings: configManager.getAllSettings()
      }, null, 2));
    } else {
      // Human-readable output
      displaySection('Ember CLI Configuration');
      
      // Display general settings
      console.log(`${chalk.bold('Default Provider:')} ${defaultProvider || chalk.dim('Not set')}`);
      console.log(`${chalk.bold('Default Model:')} ${defaultModel || chalk.dim('Not set')}`);
      console.log(`${chalk.bold('Usage Tracking:')} ${usageTracking ? chalk.green('Enabled') : chalk.yellow('Disabled')}`);
      
      // Display configured providers
      console.log(`\n${chalk.bold('Configured Providers:')}`);
      
      if (providers.length === 0) {
        console.log('  No providers configured');
      } else {
        providers.forEach(provider => {
          const isDefault = provider === defaultProvider;
          console.log(`  ${isDefault ? chalk.green('✓') : ' '} ${provider}`);
          
          // Show API key if requested
          if (options.showKeys) {
            const config = configManager.getProviderConfig(provider);
            console.log(`    API Key: ${config.apiKey}`);
          }
        });
      }
      
      // Show custom settings
      const settings = configManager.getAllSettings();
      const customSettings = Object.keys(settings).filter(key => 
        key !== 'defaultProvider' && key !== 'defaultModel' && key !== 'usageTracking'
      );
      
      if (customSettings.length > 0) {
        console.log(`\n${chalk.bold('Custom Settings:')}`);
        
        customSettings.forEach(key => {
          console.log(`  ${key}: ${JSON.stringify(settings[key])}`);
        });
      }
      
      // Show tips
      console.log('');
      displayTip(`Set a configuration value with ${chalk.cyan('ember config set <key> <value>')}`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to retrieve configuration');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Set a configuration value
 * 
 * @param key Configuration key
 * @param value Configuration value
 */
async function setConfig(key: string, value: string): Promise<void> {
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Parse value
    let parsedValue: any = value;
    
    // Try to parse as JSON if it looks like an object, array, number, or boolean
    if (value === 'true') {
      parsedValue = true;
    } else if (value === 'false') {
      parsedValue = false;
    } else if (value === 'null') {
      parsedValue = null;
    } else if (!isNaN(Number(value)) && value.trim() !== '') {
      parsedValue = Number(value);
    } else if ((value.startsWith('{') && value.endsWith('}')) || 
               (value.startsWith('[') && value.endsWith(']'))) {
      try {
        parsedValue = JSON.parse(value);
      } catch (e) {
        // Keep as string if it's not valid JSON
      }
    }
    
    // Special handling for certain keys
    if (key === 'defaultProvider') {
      configManager.setDefaultProvider(value);
    } else if (key === 'usageTracking') {
      configManager.setUsageTracking(parsedValue === true);
    } else {
      // Set as regular setting
      configManager.setSetting(key, parsedValue);
    }
    
    // Show success message
    displaySuccess(`Configuration value for ${key} set to ${JSON.stringify(parsedValue)}`);
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Get a configuration value
 * 
 * @param key Configuration key
 */
async function getConfig(key: string): Promise<void> {
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Special handling for certain keys
    let value: any;
    
    if (key === 'defaultProvider') {
      value = configManager.getDefaultProvider();
    } else if (key === 'usageTracking') {
      value = configManager.isUsageTrackingEnabled();
    } else {
      // Get as regular setting
      value = configManager.getSetting(key, null);
    }
    
    // Format and display value
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({ [key]: value }, null, 2));
    } else {
      // Human-readable output
      if (value === null || value === undefined) {
        console.log(`${chalk.bold(key)}: ${chalk.dim('Not set')}`);
      } else {
        console.log(`${chalk.bold(key)}: ${JSON.stringify(value)}`);
      }
    }
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Reset all configuration to defaults
 * 
 * @param options Command options
 */
async function resetConfig(options: any): Promise<void> {
  try {
    // Confirm reset if not forced
    if (!options.force) {
      const answers = await inquirer.prompt([{
        type: 'confirm',
        name: 'confirm',
        message: 'This will reset all configuration to defaults. Continue?',
        default: false
      }]);
      
      if (!answers.confirm) {
        console.log('Reset cancelled.');
        return;
      }
    }
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Reset configuration
    configManager.resetConfig();
    
    // Show success message
    displaySuccess('Configuration reset to defaults');
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Export configuration to a file
 * 
 * @param file File path
 */
async function exportConfig(file: string): Promise<void> {
  const spinner = ora('Exporting configuration...').start();
  
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Resolve file path
    const filePath = path.resolve(file);
    
    // Check if directory exists
    const directory = path.dirname(filePath);
    if (!fs.existsSync(directory)) {
      spinner.fail(`Directory ${directory} does not exist.`);
      return;
    }
    
    // Export configuration
    configManager.exportConfig(filePath);
    
    // Stop spinner
    spinner.stop();
    
    // Show success message
    displaySuccess(`Configuration exported to ${filePath}`);
    displayWarning('This file contains sensitive information such as API keys.');
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to export configuration');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Import configuration from a file
 * 
 * @param file File path
 * @param options Command options
 */
async function importConfig(file: string, options: any): Promise<void> {
  const spinner = ora('Importing configuration...').start();
  
  try {
    // Resolve file path
    const filePath = path.resolve(file);
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      spinner.fail(`File ${filePath} does not exist.`);
      return;
    }
    
    // Confirm import if not forced
    if (!options.force) {
      spinner.stop();
      
      const answers = await inquirer.prompt([{
        type: 'confirm',
        name: 'confirm',
        message: 'This will overwrite your current configuration. Continue?',
        default: false
      }]);
      
      if (!answers.confirm) {
        console.log('Import cancelled.');
        return;
      }
      
      spinner.start();
    }
    
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Import configuration
    configManager.importConfig(filePath);
    
    // Stop spinner
    spinner.stop();
    
    // Show success message
    displaySuccess(`Configuration imported from ${filePath}`);
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to import configuration');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Configure usage tracking
 * 
 * @param enabled Whether usage tracking is enabled
 */
async function configureUsageTracking(enabled: boolean): Promise<void> {
  try {
    // Get config manager
    const configManager = ConfigManager.getInstance();
    
    // Set usage tracking
    configManager.setUsageTracking(enabled);
    
    // Show success message
    if (enabled) {
      displaySuccess('Usage tracking enabled');
    } else {
      displaySuccess('Usage tracking disabled');
    }
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}