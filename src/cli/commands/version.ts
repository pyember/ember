/**
 * Version Command
 * 
 * Displays version information about Ember CLI and backend.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import { getPythonBridge } from '../bridge/python-bridge';
import { isJsonOutput } from '../utils/options';
import ora from 'ora';

/**
 * Register the version command with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerVersionCommand(program: Command): void {
  program
    .command('version')
    .alias('v')
    .description('Display version information')
    .option('--check', 'Check for updates')
    .action(async (options) => {
      await displayVersion(options);
    });
}

/**
 * Display version information
 * 
 * @param options Command options
 */
async function displayVersion(options: any): Promise<void> {
  // Show animated spinner
  const spinner = ora('Retrieving version information...').start();
  
  try {
    // Get Python bridge
    const bridge = getPythonBridge();
    await bridge.initialize();
    
    // Get version from backend
    const version = await bridge.getVersion();
    
    // Stop spinner
    spinner.stop();
    
    // Format and display version info
    if (isJsonOutput()) {
      // JSON output
      console.log(JSON.stringify({
        cli_version: require('../../../package.json').version,
        backend_version: version
      }, null, 2));
    } else {
      // Human-readable output
      console.log(`
${chalk.bold('Ember CLI Version:')} ${require('../../../package.json').version}
${chalk.bold('Ember Backend Version:')} ${version}
`);
    }
  } catch (error: any) {
    // Handle errors
    spinner.fail('Failed to retrieve version information');
    console.error(chalk.red('Error:'), error.message);
  }
}