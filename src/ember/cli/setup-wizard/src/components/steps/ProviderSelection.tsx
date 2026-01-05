import React from 'react';
import {Box, Text, useInput} from 'ink';
import SelectInput from 'ink-select-input';
import {LogoDisplay} from '../LogoDisplay.js';
import {Provider, PROVIDERS} from '../../types.js';

interface Props {
  onSelect: (provider: Provider) => void;
  onBack?: () => void;
  onCancel?: () => void;
}

export const ProviderSelection: React.FC<Props> = ({onSelect, onBack, onCancel}) => {
  const items = Object.entries(PROVIDERS).map(([key, info]) => {
    const desc = (info.description || '').trim();
    return {
      label: desc ? `${info.name} — ${desc}` : info.name,
      value: key as Provider,
      color: info.color,
    };
  });

  // Handle keyboard shortcuts
  useInput((input, key) => {
    if (key.escape) {
      onCancel?.();
    } else if ((input === 'b' || input === 'B')) {
      onBack?.();
    }
  });

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="cyan">
          Which AI provider would you like to use?
        </Text>
      </Box>

      <Box marginTop={2}>
        <SelectInput
          items={items}
          onSelect={(item) => onSelect(item.value)}
          indicatorComponent={() => <Text> </Text>}
          itemComponent={({label, isSelected}) => {
            const item = items.find(i => i.label === label);
            const provider = PROVIDERS[item?.value as Provider];
            return (
              <Box paddingY={1}>
                <Box>
                  <Text color="green">{isSelected ? '▸' : ' '} </Text>
                  <LogoDisplay 
                    provider={provider.id} 
                    variant="symbol"
                    size="small"
                  />
                  <Text> </Text>
                  <Text color={isSelected ? 'white' : 'gray'} bold={isSelected}>
                    {provider.name}
                  </Text>
                </Box>
              </Box>
            );
          }}
        />
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Box>
          <Text dimColor>↑↓</Text>
          <Text dimColor> - Navigate providers</Text>
        </Box>
        <Box>
          <Text dimColor>Enter</Text>
          <Text dimColor> - Select provider</Text>
        </Box>
        <Box>
          <Text dimColor>B</Text>
          <Text dimColor> - Back to setup mode</Text>
        </Box>
        <Box>
          <Text dimColor>ESC</Text>
          <Text dimColor> - Cancel setup</Text>
        </Box>
      </Box>
    </Box>
  );
};
