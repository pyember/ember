/**
 * Banner UI component for the Ember CLI
 * Displays a beautiful startup banner with the Ember logo
 */

import figlet from 'figlet';
import gradient from 'gradient-string';
import chalk from 'chalk';
import { getCliOptions } from '../utils/options';

/**
 * Displays the Ember CLI banner
 */
export function displayBanner(): void {
  const options = getCliOptions();
  
  // Skip banner in quiet mode
  if (options.quiet) return;
  
  // Create beautiful gradient
  const emberGradient = gradient(['#FF4500', '#FFA500', '#FF7F50']);
  
  // Generate the figlet text
  const figletText = figlet.textSync('Ember AI', {
    font: 'ANSI Shadow',
    horizontalLayout: 'default',
    verticalLayout: 'default',
    width: 80,
    whitespaceBreak: true
  });
  
  // Apply gradient to figlet text
  console.log(emberGradient(figletText));
  
  // Display version and tagline
  const pkg = require('../../../package.json');
  console.log(chalk.dim(`v${pkg.version} - The compositional framework for AI systems`));
  console.log('');
}