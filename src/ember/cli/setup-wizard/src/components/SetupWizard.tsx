import React, {useState} from 'react';
import {Box, Text, useApp} from 'ink';
import {Welcome} from './steps/Welcome.js';
import {SetupModeSelection} from './steps/SetupModeSelection.js';
import {ProviderSelection} from './steps/ProviderSelection.js';
import {ApiKeySetup} from './steps/ApiKeySetup.js';
import {LocalOllama} from './steps/LocalOllama.js';
import {TestConnection} from './steps/TestConnection.js';
import {Success} from './steps/Success.js';
import {ProgressIndicator} from './ProgressIndicator.js';
import {SetupState, Provider, ALL_PROVIDERS} from '../types.js';

interface Props {
  initialProvider?: Provider;
  contextModel?: string;
  skipWelcome?: boolean;
}

export const SetupWizard: React.FC<Props> = ({
  initialProvider,
  contextModel,
  skipWelcome = false
}) => {
  const {exit} = useApp();
  const [setupSuccess, setSetupSuccess] = useState(false);
  // Infer provider from context model if provided and no explicit initialProvider
  const inferredProvider: Provider | null = (() => {
    if (initialProvider) return initialProvider as Provider;
    if (!contextModel) return null;
    const id = contextModel.toLowerCase();
    if (id.startsWith('gpt') || id.startsWith('o')) return 'openai';
    if (id.startsWith('claude')) return 'anthropic';
    if (id.startsWith('gemini') || id.includes('google')) return 'google';
    if (id.startsWith('ollama') || id.startsWith('local')) return 'ollama';
    return null;
  })();
  const [state, setState] = useState<SetupState>({
    step: skipWelcome ? ((initialProvider || inferredProvider) ? ((initialProvider || inferredProvider) === 'ollama' ? 'localOllama' : 'apiKey') : 'setupMode') : 'welcome',
    provider: (initialProvider as Provider) || inferredProvider || null,
    apiKey: '',
    testResult: null,
    exampleFile: null,
    setupMode: undefined,
    configuredProviders: new Set<Provider>(),
    remainingProviders: [],
  });

  const handleSetupModeSelect = async (mode: 'single' | 'all') => {
    if (mode === 'all') {
      const nextProvider = ALL_PROVIDERS[0];
      setState({
        ...state,
        setupMode: mode,
        configuredProviders: new Set<Provider>(),
        remainingProviders: [...ALL_PROVIDERS],
        provider: nextProvider,
        step: nextProvider === 'ollama' ? 'localOllama' : 'apiKey',
      });
    } else {
      setState({...state, setupMode: mode, step: 'provider'});
    }
  };

  const handleProviderSelect = (provider: Provider) => {
    setState({...state, provider, step: provider === 'ollama' ? 'localOllama' : 'apiKey'});
  };

  const handleApiKey = (apiKey: string) => {
    setState({...state, apiKey, step: 'test'});
  };
  
  const handleBackToProviderSelection = () => {
    setState({...state, step: 'provider', apiKey: ''});
  };
  
  const handleBackToSetupMode = () => {
    setState({...state, step: 'setupMode', provider: null, apiKey: ''});
  };
  
  const handleSkipProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders) {
      const idx = state.remainingProviders.indexOf(state.provider!);
      const newRemaining = state.remainingProviders.filter((_, i) => i !== idx);
      if (newRemaining.length > 0) {
        const nextIndex = Math.min(idx, newRemaining.length - 1);
        const nextProvider = newRemaining[nextIndex];
        setState({
          ...state,
          remainingProviders: newRemaining,
          provider: nextProvider,
          apiKey: '',
          step: nextProvider === 'ollama' ? 'localOllama' : 'apiKey'
        });
      } else {
        setState({...state, remainingProviders: [], step: 'success'});
      }
    } else {
      setState({...state, step: 'success'});
    }
  };
  
  const handleCancel = () => {
    exit();
  };
  
  const handlePreviousProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders) {
      const currentIndex = state.remainingProviders.indexOf(state.provider!);
      if (currentIndex > 0) {
        const nextProvider = state.remainingProviders[currentIndex - 1];
        setState({
          ...state,
          provider: nextProvider,
          apiKey: '',
          step: nextProvider === 'ollama' ? 'localOllama' : 'apiKey'
        });
      }
    }
  };
  
  const handleNextProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders) {
      const currentIndex = state.remainingProviders.indexOf(state.provider!);
      if (currentIndex < state.remainingProviders.length - 1) {
        const nextProvider = state.remainingProviders[currentIndex + 1];
        setState({
          ...state,
          provider: nextProvider,
          apiKey: '',
          step: nextProvider === 'ollama' ? 'localOllama' : 'apiKey'
        });
      }
    }
  };

  const handleTestComplete = (success: boolean, message: string, exampleFile?: string) => {
    if (success) {
      const newConfigured = new Set(state.configuredProviders);
      if (state.provider) {
        newConfigured.add(state.provider);
      }
      
      // Check if we're in multi-provider mode and have more to configure
      if (state.setupMode === 'all' && state.remainingProviders && state.remainingProviders.length > 1) {
        const remaining = state.remainingProviders.slice(1);
        setState({
          ...state,
          configuredProviders: newConfigured,
          remainingProviders: remaining,
          provider: remaining[0],
          apiKey: '',
          step: 'apiKey'
        });
      } else {
        setState({...state, configuredProviders: newConfigured, exampleFile: exampleFile || 'hello_ember.py', testResult: {success, message}, step: 'success'});
      }
    } else {
      setState({...state, testResult: {success, message}, step: 'apiKey'});
    }
  };

  const handleComplete = () => {
    exit();
  };

  return (
    <Box flexDirection="column" paddingY={1}>
      {/* Progress indicator for multi-provider setup */}
      {state.setupMode === 'all' && state.step !== 'welcome' && state.step !== 'setupMode' && (
        <Box marginBottom={2}>
          <ProgressIndicator 
            configuredProviders={state.configuredProviders || new Set()}
            currentProvider={state.provider}
          />
        </Box>
      )}
      
      {state.step === 'welcome' && <Welcome onNext={() => setState({...state, step: 'setupMode'})} />}
      {state.step === 'setupMode' && <SetupModeSelection onSelectMode={handleSetupModeSelect} />}
      {state.step === 'provider' && (
        <ProviderSelection 
          onSelect={handleProviderSelect}
          onBack={handleBackToSetupMode}
          onCancel={handleCancel}
        />
      )}
      {state.step === 'apiKey' && (
        <ApiKeySetup 
          provider={state.provider!} 
          onSubmit={handleApiKey}
          onSkip={handleSkipProvider}
          onCancel={handleCancel}
          onPrevious={handlePreviousProvider}
          onNext={handleNextProvider}
          onBack={state.setupMode === 'single' ? handleBackToProviderSelection : undefined}
          error={state.testResult?.success === false ? state.testResult.message : undefined}
          isFirstProvider={state.setupMode === 'all' && state.remainingProviders ? state.remainingProviders.indexOf(state.provider!) === 0 : undefined}
          isLastProvider={state.setupMode === 'all' && state.remainingProviders ? state.remainingProviders.indexOf(state.provider!) === state.remainingProviders.length - 1 : undefined}
          setupMode={state.setupMode}
        />
      )}
      {state.step === 'localOllama' && (
        <LocalOllama 
          onBack={state.setupMode === 'single' ? handleBackToProviderSelection : undefined}
          onSkip={handleSkipProvider}
          onCancel={handleCancel}
          onComplete={(exampleFile?: string) => setState({...state, exampleFile: exampleFile || 'hello_ember.py', step: 'success'})}
        />
      )}
      {state.step === 'test' && <TestConnection provider={state.provider!} apiKey={state.apiKey} onComplete={handleTestComplete} />}
      {state.step === 'success' && <Success provider={state.provider!} onComplete={handleComplete} configuredProviders={state.configuredProviders} exampleFile={state.exampleFile || 'hello_ember.py'} />}
    </Box>
  );
};
