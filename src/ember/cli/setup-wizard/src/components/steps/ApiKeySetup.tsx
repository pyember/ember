import React, {useState, useEffect} from 'react';
import {Box, Text, useInput} from 'ink';
import TextInput from 'ink-text-input';
import {Link} from '../Link.js';
import {Provider, PROVIDERS} from '../../types.js';
import clipboardy from 'clipboardy';

interface Props {
  provider: Provider;
  onSubmit: (apiKey: string) => void;
  onSkip?: () => void;
  onCancel?: () => void;
  onPrevious?: () => void;
  onNext?: () => void;
  onBack?: () => void;
  error?: string;
  isFirstProvider?: boolean;
  isLastProvider?: boolean;
  setupMode?: 'single' | 'all';
}

export const ApiKeySetup: React.FC<Props> = ({provider, onSubmit, onSkip, onCancel, onPrevious, onNext, onBack, error, isFirstProvider, isLastProvider, setupMode}) => {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [clipboardStatus, setClipboardStatus] = useState('');
  const [localWarning, setLocalWarning] = useState('');
  const providerInfo = PROVIDERS[provider];

  // Handle keyboard input for Tab and clipboard paste
  useInput((input, key) => {
    if (key.tab) {
      // Toggle visibility
      setShowKey(prev => !prev);
    } else if (input === 'p' || input === 'P') {
      // Paste from clipboard
      handleClipboardPaste();
    } else if (key.escape) {
      // Cancel setup
      onCancel?.();
    } else if ((input === 'b' || input === 'B') && setupMode === 'single') {
      // Go back to provider selection (only in single provider mode)
      onBack?.();
    } else if ((input === 's' || input === 'S') && setupMode === 'all') {
      // Skip this provider (only in multi-provider mode)
      onSkip?.();
    } else if (key.leftArrow && !isFirstProvider && setupMode === 'all') {
      // Go to previous provider (only in multi-provider mode)
      onPrevious?.();
    } else if (key.rightArrow && !isLastProvider && setupMode === 'all') {
      // Go to next provider (only in multi-provider mode)
      onNext?.();
    }
  });

  const handleClipboardPaste = async () => {
    try {
      const text = await clipboardy.read();
      if (text && text.trim()) {
        setApiKey(text.trim());
        setClipboardStatus('✓ Pasted from clipboard');
        setTimeout(() => setClipboardStatus(''), 2000);
      }
    } catch (err) {
      setClipboardStatus('✗ Could not read clipboard');
      setTimeout(() => setClipboardStatus(''), 2000);
    }
  };

  const handleSubmit = () => {
    const trimmed = apiKey.trim();
    if (trimmed.includes(' ')) {
      setLocalWarning('API key should not contain spaces');
      return;
    }
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
      setLocalWarning('Remove surrounding quotes from the API key');
      return;
    }
    if (trimmed.length < 8) {
      setLocalWarning('API key looks too short');
      return;
    }
    setLocalWarning('');
    if (trimmed) {
      onSubmit(apiKey.trim());
    }
  };

  const maskApiKey = (key: string) => {
    if (!key || key.length < 8) return key;
    if (showKey) return key;
    return key.slice(0, 3) + '•'.repeat(key.length - 6) + key.slice(-3);
  };

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Configure your {providerInfo.name} API key
        </Text>
      </Box>

      <Box marginBottom={2}>
        <Text>Get your API key from: </Text>
        <Link url={providerInfo.keyUrl}>
          <Text color="blue" underline>
            {providerInfo.keyUrl}
          </Text>
        </Link>
      </Box>

      {error && (
        <Box marginBottom={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}
      {localWarning && (
        <Box marginBottom={1}>
          <Text color="yellow">{localWarning}</Text>
        </Box>
      )}

      <Box>
        <Text>API Key: </Text>
        <Box flexDirection="column">
          <TextInput
            value={apiKey}
            onChange={setApiKey}
            onSubmit={handleSubmit}
            placeholder="Paste your API key here..."
            mask={showKey ? undefined : '•'}
          />
          <Box marginTop={1}>
            <Text dimColor>
              {showKey ? 'Visible' : 'Hidden'}: {maskApiKey(apiKey)}
            </Text>
          </Box>
          {clipboardStatus && (
            <Box marginTop={1}>
              <Text color="green">{clipboardStatus}</Text>
            </Box>
          )}
        </Box>
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Box>
          <Text dimColor>Tab</Text>
          <Text dimColor> - Toggle visibility ({showKey ? 'visible' : 'hidden'})</Text>
        </Box>
        <Box>
          <Text dimColor>P</Text>
          <Text dimColor> - Paste from clipboard</Text>
        </Box>
        {setupMode === 'single' && (
          <Box>
            <Text dimColor>B</Text>
            <Text dimColor> - Back to provider selection</Text>
          </Box>
        )}
        {setupMode === 'all' && (
          <>
            <Box>
              <Text dimColor>S</Text>
              <Text dimColor> - Skip this provider</Text>
            </Box>
            {!isFirstProvider && (
              <Box>
                <Text dimColor>←</Text>
                <Text dimColor> - Previous provider</Text>
              </Box>
            )}
            {!isLastProvider && (
              <Box>
                <Text dimColor>→</Text>
                <Text dimColor> - Next provider</Text>
              </Box>
            )}
          </>
        )}
        <Box>
          <Text dimColor>Enter</Text>
          <Text dimColor> - {setupMode === 'single' ? 'Save and finish' : 'Continue'}</Text>
        </Box>
        <Box>
          <Text dimColor>ESC</Text>
          <Text dimColor> - Cancel setup</Text>
        </Box>
      </Box>

      <Box marginTop={1}>
        <Text dimColor italic>
          Your key will be saved securely to ~/.ember/credentials
        </Text>
      </Box>
    </Box>
  );
};
