import React from 'react';
import { Box, Text } from 'ink';
import { Provider, PROVIDERS, ALL_PROVIDERS } from '../types.js';

interface ProgressIndicatorProps {
  configuredProviders: Set<Provider>;
  currentProvider: Provider | null;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ 
  configuredProviders, 
  currentProvider 
}) => {
  const allProviders: Provider[] = ALL_PROVIDERS;
  const progress = Math.round((configuredProviders.size / allProviders.length) * 100);
  
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text dimColor>Configuring providers ({configuredProviders.size}/{allProviders.length})</Text>
      </Box>
      <Box marginTop={1} flexDirection="row">
        {allProviders.map((provider, index) => {
          const isConfigured = configuredProviders.has(provider);
          const isCurrent = provider === currentProvider;
          return (
            <React.Fragment key={provider}>
              {index > 0 && <Text>  </Text>}
              <Box flexDirection="row">
                <Text color={isConfigured ? 'green' : isCurrent ? 'yellow' : 'gray'}>
                  {isConfigured ? '✓' : isCurrent ? '○' : '○'}
                </Text>
                <Text> </Text>
                <Text color={isConfigured ? 'green' : isCurrent ? 'yellow' : 'gray'}>
                  {PROVIDERS[provider].name}
                </Text>
              </Box>
            </React.Fragment>
          );
        })}
      </Box>
      <Box marginTop={1}>
        <Text dimColor>[</Text>
        <Text color="green">{'█'.repeat(Math.floor(progress / 10))}</Text>
        <Text dimColor>{'░'.repeat(10 - Math.floor(progress / 10))}</Text>
        <Text dimColor>] {progress}%</Text>
      </Box>
    </Box>
  );
};
