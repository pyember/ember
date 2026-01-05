export type Provider = 'openai' | 'anthropic' | 'google' | 'ollama';

// Single source of truth for provider ordering across the wizard
export const ALL_PROVIDERS: Provider[] = ['openai', 'anthropic', 'google', 'ollama'];

export interface ProviderInfo {
  id: Provider;
  name: string;
  models: string[];
  description: string;
  keyUrl: string;
  testModel: string;
  icon: string;
  color?: string; // Terminal color for the logo
}

export interface SetupState {
  step: 'welcome' | 'setupMode' | 'provider' | 'apiKey' | 'localOllama' | 'test' | 'success';
  provider: Provider | null;
  apiKey: string;
  testResult: {success: boolean; message: string} | null;
  exampleFile: string | null;
  // Multi-provider setup support
  setupMode?: 'single' | 'all';
  configuredProviders?: Set<Provider>;
  remainingProviders?: Provider[];
}

export const PROVIDERS: Record<Provider, ProviderInfo> = {
  openai: {
    id: 'openai',
    name: 'OpenAI',
    models: ['GPT-4', 'GPT-3.5'],
    description: 'Popular general-purpose models',
    keyUrl: 'https://platform.openai.com/settings/organization/api-keys',
    testModel: 'gpt-3.5-turbo',
    icon: '◯',
    color: 'green',
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic',
    models: ['Claude 3 Opus', 'Claude 3 Sonnet'],
    description: 'Strong on reasoning and safety',
    keyUrl: 'https://console.anthropic.com/settings/keys',
    testModel: 'claude-3-haiku',
    icon: '△',
    color: 'yellow',
  },
  google: {
    id: 'google',
    name: 'Google',
    models: ['Gemini 1.5 Pro', 'Gemini 1.5 Flash'],
    description: 'Latest Gemini 1.5 multimodal models',
    keyUrl: 'https://aistudio.google.com/apikey',
    testModel: 'gemini-1.5-pro-latest',
    icon: 'G',
    color: 'blue',
  },
  ollama: {
    id: 'ollama',
    name: 'Ollama (Local)',
    models: ['llama3.2:1b', 'qwen2.5:0.5b', 'tinyllama'],
    description: 'Run local models (no API key)',
    keyUrl: 'https://ollama.com',
    testModel: 'llama3.2:1b',
    icon: '◎',
    color: 'magenta',
  },
};
