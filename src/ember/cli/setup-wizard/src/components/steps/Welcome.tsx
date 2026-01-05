import React from 'react';
import {Box, Text, useInput} from 'ink';
import {CubeAnimation} from '../CubeAnimation.js';

interface Props {
  onNext: () => void;
}

export const Welcome: React.FC<Props> = ({onNext}) => {
  useInput(() => {
    onNext();
  });

  return (
    <Box flexDirection="column" alignItems="center" paddingY={1}>
      <Box marginBottom={1}>
        <CubeAnimation />
      </Box>
      <Box marginBottom={0}>
        <Text color="blue" bold>
          EMBER
        </Text>
      </Box>
      <Box marginBottom={2}>
        <Text dimColor>
          Inference-time scaling and compound agent systems
        </Text>
      </Box>
      <Box>
        <Text dimColor italic>
          Press any key to continue...
        </Text>
      </Box>
    </Box>
  );
};