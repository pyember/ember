/**
 * Project Commands
 * 
 * Commands for creating and managing Ember projects.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import inquirer from 'inquirer';
import ora from 'ora';
import fs from 'fs';
import path from 'path';
import { Listr2 } from 'listr2';

import { getPythonBridge } from '../bridge/python-bridge';
import { isJsonOutput } from '../utils/options';
import { displaySection, displaySuccess, displayTip } from '../ui/intro';

/**
 * Register project commands with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerProjectCommands(program: Command): void {
  const projectCommand = program
    .command('project')
    .description('Manage Ember projects');
  
  // Create a new project
  projectCommand
    .command('new <name>')
    .description('Create a new Ember project')
    .option('-t, --template <template>', 'Project template', 'basic')
    .option('-d, --directory <directory>', 'Project directory (defaults to name)')
    .option('-p, --provider <provider>', 'Default provider to use')
    .option('-m, --model <model>', 'Default model to use')
    .action(async (name, options) => {
      await createProject(name, options);
    });
  
  // List available project templates
  projectCommand
    .command('templates')
    .description('List available project templates')
    .action(async () => {
      await listTemplates();
    });
  
  // Analyze project
  projectCommand
    .command('analyze [directory]')
    .description('Analyze an Ember project')
    .action(async (directory = '.') => {
      await analyzeProject(directory);
    });
}

/**
 * Create a new Ember project
 * 
 * @param name Project name
 * @param options Command options
 */
async function createProject(name: string, options: any): Promise<void> {
  // Validate name
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    console.error(chalk.red('Error:'), 'Project name must contain only letters, numbers, underscores, and hyphens.');
    return;
  }
  
  // Get directory
  const directory = options.directory || name;
  const fullPath = path.resolve(directory);
  
  // Check if directory exists
  if (fs.existsSync(fullPath)) {
    const answers = await inquirer.prompt([{
      type: 'confirm',
      name: 'overwrite',
      message: `Directory ${directory} already exists. Continue anyway?`,
      default: false
    }]);
    
    if (!answers.overwrite) {
      console.log('Project creation cancelled.');
      return;
    }
  }
  
  // Initialize tasks
  const tasks = new Listr2([
    {
      title: 'Initializing project',
      task: async (ctx, task) => {
        const bridge = getPythonBridge();
        await bridge.initialize();
        
        // Create options for project creation
        const projectOptions = {
          template: options.template || 'basic',
          provider: options.provider,
          model: options.model
        };
        
        // Create project
        await bridge.createProject(directory, projectOptions);
      }
    },
    {
      title: 'Installing dependencies',
      task: async (ctx, task) => {
        // Simulate installation
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    },
    {
      title: 'Setting up configuration',
      task: async (ctx, task) => {
        // Simulate configuration
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    },
    {
      title: 'Finalizing project',
      task: async (ctx, task) => {
        // Simulate finalization
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
  ]);
  
  try {
    // Execute tasks
    await tasks.run();
    
    // Show success message
    if (!isJsonOutput()) {
      displaySuccess(`Project ${name} created successfully in ${directory}`);
      console.log('');
      console.log(`To get started, run:`);
      console.log(chalk.cyan(`  cd ${directory}`));
      console.log(chalk.cyan('  pip install -e .'));
      console.log('');
      displayTip('Set up your API keys as environment variables or configure them with ember provider configure <provider>');
    } else {
      console.log(JSON.stringify({
        status: 'success',
        project: {
          name,
          directory: fullPath,
          template: options.template || 'basic'
        }
      }, null, 2));
    }
  } catch (error: any) {
    // Handle errors
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * List available project templates
 */
async function listTemplates(): Promise<void> {
  const spinner = ora('Retrieving templates...').start();
  
  try {
    // Define templates (simulated - would be fetched from backend)
    const templates = [
      {
        id: 'basic',
        name: 'Basic Project',
        description: 'Simple starter project with minimal dependencies',
        features: ['Direct model invocation', 'Basic usage examples']
      },
      {
        id: 'complete',
        name: 'Complete Project',
        description: 'Full-featured project with all operators and utilities',
        features: ['Operators', 'Evaluation tools', 'Configuration examples', 'Advanced usage patterns']
      },
      {
        id: 'api',
        name: 'API Project',
        description: 'Project template for building APIs with Ember',
        features: ['FastAPI integration', 'API endpoint examples', 'Authentication boilerplate']
      },
      {
        id: 'notebook',
        name: 'Notebook Project',
        description: 'Jupyter notebook-based project for experimentation',
        features: ['Jupyter notebooks', 'Example experiments', 'Visualization tools']
      }
    ];
    
    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Stop spinner
    spinner.stop();
    
    // Format and display templates
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({ templates }, null, 2));
    } else {
      // Human-readable output
      displaySection('Available Project Templates');
      
      templates.forEach(template => {
        console.log('');
        console.log(`${chalk.bold.green(template.name)} ${chalk.gray(`(${template.id})`)}`);
        console.log(chalk.dim(template.description));
        
        console.log(chalk.bold('\nFeatures:'));
        template.features.forEach(feature => {
          console.log(`  • ${feature}`);
        });
        
        console.log(chalk.dim('─'.repeat(50)));
      });
      
      // Show tip
      displayTip(`Create a project with ${chalk.cyan('ember project new <name> --template <template>')}`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to retrieve templates');
    console.error(chalk.red('Error:'), error.message);
  }
}

/**
 * Analyze an Ember project
 * 
 * @param directory Project directory
 */
async function analyzeProject(directory: string): Promise<void> {
  const spinner = ora(`Analyzing project in ${directory}...`).start();
  
  try {
    // Check if directory exists
    const fullPath = path.resolve(directory);
    
    if (!fs.existsSync(fullPath)) {
      spinner.fail(`Directory ${directory} does not exist.`);
      return;
    }
    
    // Check if it's an Ember project
    const isEmberProject = fs.existsSync(path.join(fullPath, 'ember_example.py')) || 
                          fs.existsSync(path.join(fullPath, 'pyproject.toml')); // Simple heuristic
    
    if (!isEmberProject) {
      spinner.fail(`Directory ${directory} does not appear to be an Ember project.`);
      return;
    }
    
    // Simulate analysis
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Create fake analysis results
    const analysis = {
      project: {
        name: path.basename(fullPath),
        directory: fullPath,
        type: 'Basic Project',
      },
      structure: {
        operators: 2,
        models: 1,
        data_sources: 1,
        test_files: 3
      },
      dependencies: {
        required: ['ember-ai', 'openai', 'pandas'],
        missing: []
      },
      config: {
        providers: ['openai'],
        configured: true
      },
      suggestions: [
        'Add automated tests for your operators',
        'Consider implementing usage tracking',
        'Update to the latest Ember version'
      ]
    };
    
    // Stop spinner
    spinner.stop();
    
    // Format and display analysis
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify(analysis, null, 2));
    } else {
      // Human-readable output
      displaySection(`Project Analysis: ${analysis.project.name}`);
      
      console.log(`${chalk.bold('Project Type:')} ${analysis.project.type}`);
      console.log(`${chalk.bold('Directory:')} ${analysis.project.directory}`);
      
      // Display structure
      console.log(`\n${chalk.bold('Project Structure:')}`);
      console.log(`  Operators: ${analysis.structure.operators}`);
      console.log(`  Models: ${analysis.structure.models}`);
      console.log(`  Data Sources: ${analysis.structure.data_sources}`);
      console.log(`  Test Files: ${analysis.structure.test_files}`);
      
      // Display dependencies
      console.log(`\n${chalk.bold('Dependencies:')}`);
      console.log(`  Required: ${analysis.dependencies.required.join(', ')}`);
      
      if (analysis.dependencies.missing.length > 0) {
        console.log(`  ${chalk.red('Missing:')} ${analysis.dependencies.missing.join(', ')}`);
      } else {
        console.log(`  ${chalk.green('All dependencies satisfied')}`);
      }
      
      // Display config
      console.log(`\n${chalk.bold('Configuration:')}`);
      console.log(`  Providers: ${analysis.config.providers.join(', ')}`);
      console.log(`  Status: ${analysis.config.configured ? chalk.green('Configured') : chalk.yellow('Not Configured')}`);
      
      // Display suggestions
      console.log(`\n${chalk.bold('Suggestions:')}`);
      analysis.suggestions.forEach(suggestion => {
        console.log(`  • ${suggestion}`);
      });
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail(`Failed to analyze project in ${directory}`);
    console.error(chalk.red('Error:'), error.message);
  }
}