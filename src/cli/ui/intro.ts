/**
 * Intro UI component for the Ember CLI
 * Displays helpful introduction text for commands
 */

import chalk from 'chalk';
import emoji from 'node-emoji';
import terminalLink from 'terminal-link';
import { getCliOptions } from '../utils/options';

/**
 * Displays the command introduction text
 */
export function displayIntro(): void {
  const options = getCliOptions();
  
  // Skip intro in quiet mode
  if (options.quiet) return;
  
  // Display helpful text
  console.log(`${emoji.get('fire')} ${chalk.bold('Welcome to the Ember AI CLI!')}`);
  console.log('');
  
  // Create links
  const docsLink = terminalLink('Documentation', 'https://docs.pyember.org');
  const githubLink = terminalLink('GitHub', 'https://github.com/pyember/ember');
  
  // Show helpful links
  console.log(`${emoji.get('book')} ${docsLink}  ${emoji.get('computer')} ${githubLink}`);
  console.log('');
}

/**
 * Display a styled section header
 * 
 * @param title The section title
 */
export function displaySection(title: string): void {
  const options = getCliOptions();
  
  // Skip in quiet mode
  if (options.quiet) return;
  
  console.log(chalk.bold.yellow(`\n▸ ${title}`));
}

/**
 * Display a success message
 * 
 * @param message The success message
 */
export function displaySuccess(message: string): void {
  console.log(`${emoji.get('white_check_mark')} ${chalk.green(message)}`);
}

/**
 * Display a warning message
 * 
 * @param message The warning message
 */
export function displayWarning(message: string): void {
  console.log(`${emoji.get('warning')} ${chalk.yellow(message)}`);
}

/**
 * Display a tip message
 * 
 * @param message The tip message
 */
export function displayTip(message: string): void {
  const options = getCliOptions();
  
  // Skip in quiet mode
  if (options.quiet) return;
  
  console.log(`${emoji.get('bulb')} ${chalk.cyan('Tip:')} ${message}`);
}