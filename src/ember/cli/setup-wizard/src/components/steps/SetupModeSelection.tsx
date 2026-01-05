import React from 'react';
import { Box, Text } from 'ink';
import SelectInput from 'ink-select-input';

interface SetupModeSelectionProps {
  onSelectMode: (mode: 'single' | 'all') => void;
}

export const SetupModeSelection: React.FC<SetupModeSelectionProps> = ({ onSelectMode }) => {
  const items = [
    { label: 'Configure a single provider', value: 'single' },
    { label: 'Configure all providers', value: 'all' }
  ];

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold>How would you like to set up Ember?</Text>
      </Box>
      
      <SelectInput
        items={items}
        onSelect={(item) => onSelectMode(item.value as 'single' | 'all')}
      />
    </Box>
  );
};