# Ember Setup Wizard

A delightful, interactive setup experience for Ember that gets users from zero to their first API call in under 60 seconds.

## Usage

```bash
npx @ember-ai/setup
```

No installation required - npx runs it directly!

## Features

- Terminal UI with clear, accessible prompts
- Fast setup flow
- Secure API key configuration
- Automatic connection testing
- Working example generation
- Provider recommendations

## Development

```bash
# Install dependencies
npm install

# Run in development
npm run dev

# Build for production
npm run build
```

## Design Philosophy

This setup wizard is designed around a short, reliable path to first success:

1. **Zero Friction** - Works immediately with `npx`, no installation
2. **Progressive Disclosure** - Shows only what's needed at each step
3. **Instant Feedback** - Visual confirmation at every action
4. **Error Recovery** - Clear guidance when things go wrong
5. **Polish** - Small UX details that reduce confusion

## Technical Stack

- **Ink** - React for CLIs
- **TypeScript** - Type safety
- **ink-gradient** - Beautiful text effects
- **ink-spinner** - Smooth loading states
- **ink-select-input** - Intuitive selection
- **ink-text-input** - Secure input handling

## Why NPM?

While Ember is a Python library, we chose npm for the setup wizard because:

1. **Superior Terminal UX** - React-based Ink provides unmatched CLI experiences
2. **No Python Dependencies** - Users might not have Python configured yet
3. **Instant Start** - npx requires no installation
4. **Async Excellence** - Better for API calls and progress indication
5. **UX tooling** - The npm ecosystem enables a refined terminal experience

The goal is a smooth onboarding experience with minimal local prerequisites.
