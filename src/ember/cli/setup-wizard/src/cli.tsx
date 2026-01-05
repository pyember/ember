#!/usr/bin/env node
import React from 'react';
import {render} from 'ink';
import {SetupWizard} from './components/SetupWizard.js';

// Check if launched from missing key context
const context = process.env.EMBER_SETUP_CONTEXT;
const provider = process.env.EMBER_SETUP_PROVIDER;
const model = process.env.EMBER_SETUP_MODEL;

// Only clear if not in missing-key context (to preserve the prompt)
if (context !== 'missing-key') {
  console.clear();
}

// Handle unhandled rejections gracefully
process.on('unhandledRejection', (error) => {
  console.error('\nSetup failed:', error);
  process.exit(1);
});

process.on('uncaughtException', (error) => {
  console.error('\nSetup failed:', error);
  process.exit(1);
});

// Pass context to wizard and handle exit
const {waitUntilExit} = render(<SetupWizard 
  initialProvider={provider as any}
  contextModel={model}
  skipWelcome={context === 'missing-key'}
/>);

waitUntilExit().then(() => {
  process.exit(0);
}).catch(() => {
  process.exit(1);
});