#!/usr/bin/env node

/**
 * Demo script to showcase the various progress indicators
 * Run with: npx ts-node src/cli/ui/demo-progress.ts
 */

import chalk from 'chalk';
import { 
  AnimatedSpinner,
  SpinnerStyle,
  createThinkingSpinner,
  createEmberSpinner,
  AnimatedProgress,
  ProgressBarStyle,
  createDownloadProgress,
  createEmberProgress
} from './index';

// Helper to pause execution
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function demonstrateSpinners() {
  console.log(chalk.bold.yellow('\n=== Spinner Styles Demo ===\n'));
  
  // Classic spinner
  console.log(chalk.cyan('Classic spinner:'));
  const classicSpinner = new AnimatedSpinner({ 
    text: 'Loading data...',
    style: SpinnerStyle.Classic
  });
  classicSpinner.start();
  await sleep(3000);
  classicSpinner.succeed('Data loaded successfully');
  
  // Dots spinner
  console.log(chalk.cyan('\nDots spinner:'));
  const dotsSpinner = new AnimatedSpinner({ 
    text: 'Connecting to server...',
    style: SpinnerStyle.Dots,
    color: 'blue'
  });
  dotsSpinner.start();
  await sleep(3000);
  dotsSpinner.succeed('Connected to server');
  
  // Pulse spinner
  console.log(chalk.cyan('\nPulse spinner:'));
  const pulseSpinner = new AnimatedSpinner({ 
    text: 'Processing request...',
    style: SpinnerStyle.Pulse,
    color: 'magenta'
  });
  pulseSpinner.start();
  await sleep(3000);
  pulseSpinner.fail('Failed to process request');
  
  // Ember logo spinner
  console.log(chalk.cyan('\nEmber logo spinner:'));
  const emberSpinner = createEmberSpinner('Initializing Ember system...');
  emberSpinner.start();
  await sleep(4000);
  emberSpinner.succeed('Ember system initialized');
  
  // Thinking spinner with keywords
  console.log(chalk.cyan('\nThinking spinner with keywords:'));
  const thinkingSpinner = createThinkingSpinner();
  thinkingSpinner.start();
  await sleep(8000);
  thinkingSpinner.succeed('Processing complete');
}

async function demonstrateProgressBars() {
  console.log(chalk.bold.yellow('\n=== Progress Bar Styles Demo ===\n'));
  
  // Standard progress bar
  console.log(chalk.cyan('Standard progress bar:'));
  const standardProgress = new AnimatedProgress({
    total: 100,
    style: ProgressBarStyle.Standard,
    text: 'Processing items',
    autoStart: true
  });
  
  for (let i = 0; i <= 100; i += 5) {
    standardProgress.update(i);
    await sleep(100);
  }
  standardProgress.complete('Processing complete');
  
  // Blocks progress bar
  console.log(chalk.cyan('\nBlocks progress bar:'));
  const blocksProgress = new AnimatedProgress({
    total: 100,
    style: ProgressBarStyle.Blocks,
    color: 'green',
    text: 'Installing dependencies',
    autoStart: true
  });
  
  for (let i = 0; i <= 100; i += 2) {
    blocksProgress.update(i);
    await sleep(50);
  }
  blocksProgress.complete('Dependencies installed');
  
  // Simple progress bar
  console.log(chalk.cyan('\nSimple progress bar:'));
  const simpleProgress = new AnimatedProgress({
    total: 100,
    style: ProgressBarStyle.Simple,
    color: 'blue',
    text: 'Building project',
    showEta: true,
    autoStart: true
  });
  
  for (let i = 0; i <= 100; i += 5) {
    simpleProgress.update(i);
    await sleep(150);
  }
  simpleProgress.complete('Project built successfully');
  
  // Rainbow progress bar
  console.log(chalk.cyan('\nRainbow progress bar:'));
  const rainbowProgress = new AnimatedProgress({
    total: 100,
    style: ProgressBarStyle.Rainbow,
    text: 'Generating awesome',
    autoStart: true
  });
  
  for (let i = 0; i <= 100; i += 1) {
    rainbowProgress.update(i);
    await sleep(30);
  }
  rainbowProgress.complete('Awesome generated!');
  
  // Download progress bar
  console.log(chalk.cyan('\nDownload progress bar:'));
  const downloadProgress = createDownloadProgress(1024 * 1024 * 15, 'Downloading package');
  
  let downloaded = 0;
  const chunkSize = 1024 * 100; // 100KB chunks
  
  while (downloaded < downloadProgress.total) {
    // Simulate variable download speeds
    const randomChunk = chunkSize * (0.5 + Math.random());
    downloaded = Math.min(downloaded + randomChunk, downloadProgress.total);
    downloadProgress.update(downloaded);
    await sleep(100);
  }
  downloadProgress.complete('Package downloaded');
  
  // Ember progress bar
  console.log(chalk.cyan('\nEmber progress bar:'));
  const emberProgress = createEmberProgress(100, 'Building Ember app');
  
  for (let i = 0; i <= 100; i += 2) {
    emberProgress.update(i);
    await sleep(70);
  }
  emberProgress.complete('Ember app built successfully');
}

async function main() {
  console.log(chalk.bold.magenta('=== Ember CLI Progress Indicators Demo ==='));
  
  await demonstrateSpinners();
  await demonstrateProgressBars();
  
  console.log(chalk.bold.green('\n✅ Demo completed successfully!\n'));
}

main().catch(console.error);