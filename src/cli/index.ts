#!/usr/bin/env node

/**
 * Ember CLI - A beautiful and robust command line interface for Ember AI
 * 
 * This CLI follows SOLID principles:
 * - Single Responsibility: Each module has one job
 * - Open/Closed: Extensible without modification
 * - Liskov Substitution: Interface implementations are interchangeable
 * - Interface Segregation: Small, specific interfaces
 * - Dependency Inversion: High-level modules don't depend on low-level ones
 */

import { Command } from 'commander';
import updateNotifier from 'update-notifier';
import chalk from 'chalk';

// Import commands
import { registerVersionCommand } from './commands/version';
import { registerProviderCommands } from './commands/provider';
import { registerModelCommands } from './commands/model';
import { registerProjectCommands } from './commands/project';
import { registerInvokeCommand } from './commands/invoke';
import { registerConfigCommands } from './commands/config';
import { registerCompletionCommand } from './utils/completion';

// Import UI components
import { displayBanner } from './ui/banner';
import { displayIntro } from './ui/intro';

// Import error handling utilities
import { handleError, EmberCliError } from './errors';
import { cleanupPythonBridge } from './bridge/python-bridge';

// Import package.json
import pkg from '../../package.json';

/**
 * Main CLI entry point
 */
async function main() {
  try {
    // Check for updates
    updateNotifier({ pkg }).notify();

    // Show welcome banner
    displayBanner();
    
    // Create program
    const program = new Command()
      .name('ember')
      .description('Ember AI CLI - Build and orchestrate compound AI systems')
      .version(pkg.version, '-v, --version', 'Display CLI version')
      .option('--no-color', 'Disable colors')
      .option('--debug', 'Enable debug mode')
      .option('--json', 'Output as JSON where applicable')
      .option('--quiet', 'Suppress non-essential output')
      .option('--compact', 'Display compact output')
      .option('--verbose', 'Display verbose output')
      .hook('preAction', (thisCommand, actionCommand) => {
        // Skip intro for version command or when quiet is set
        const isQuiet = thisCommand.opts().quiet || actionCommand.name() === 'version';
        if (!isQuiet && actionCommand.parent?.name() === 'ember') {
          displayIntro();
        }
      });

    // Register error handling for command errors
    program.configureOutput({
      outputError: (str, write) => {
        // Create a generic error from the commander error string
        const error = new EmberCliError(str.trim());
        handleError(error, {}, true);
      }
    });

    // Register commands
    registerVersionCommand(program);
    registerProviderCommands(program);
    registerModelCommands(program);
    registerProjectCommands(program);
    registerInvokeCommand(program);
    registerConfigCommands(program);
    registerCompletionCommand(program);

    // Parse arguments or show help
    program.parse(process.argv);

    // If no arguments, show help
    if (process.argv.length <= 2) {
      program.help();
    }
  } catch (error) {
    handleError(error, {}, true);
  }
}

// Handle errors
process.on('uncaughtException', (err) => {
  handleError(err, { includeStack: true }, true);
});

process.on('unhandledRejection', (reason: any) => {
  handleError(reason, { includeStack: true }, true);
});

// Clean up resources on exit signals
process.on('SIGINT', async () => {
  console.log('\nInterrupted. Cleaning up...');
  await cleanupPythonBridge();
  process.exit(130); // Standard exit code for SIGINT
});

process.on('SIGTERM', async () => {
  console.log('\nTerminated. Cleaning up...');
  await cleanupPythonBridge();
  process.exit(143); // Standard exit code for SIGTERM
});

// Execute the main function
main().catch(err => {
  handleError(err, {}, true);
});