import React from 'react';
import { Box, Text } from 'ink';
import { getLogo, supportsUnicode } from '../logos/index.js';

interface LogoDisplayProps {
  provider?: string;
  variant?: 'symbol' | 'full' | 'ascii';
  size?: 'small' | 'medium' | 'large';
}

const ASCII_LOGOS: Record<string, string[]> = {
  ember: [
    '    ▄▄▄▄▄▄▄▄▄▄▄▄    ',
    '  ▄█▓▓▓▓▓▓▓▓▓▓▓▓█▄  ',
    ' █▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█ ',
    '█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█',
    ' █▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█ ',
    '  ▀█▓▓▓▓▓▓▓▓▓▓▓▓█▀  ',
    '    ▀▀▀▀▀▀▀▀▀▀▀▀    '
  ],
  openai: [
    '    ╭─────────╮    ',
    '   ╱           ╲   ',
    '  │      ●      │  ',
    '  │             │  ',
    '   ╲           ╱   ',
    '    ╰─────────╯    '
  ],
  anthropic: [
    '       ▲       ',
    '      ╱ ╲      ',
    '     ╱   ╲     ',
    '    ╱     ╲    ',
    '   ╱       ╲   ',
    '  ╱─────────╲  '
  ],
  google: [
    '    ╭───────╮    ',
    '   ╱         ╲   ',
    '  │     G     │  ',
    '  │           │  ',
    '   ╲         ╱   ',
    '    ╰───────╯    '
  ]
};

const SMALL_LOGOS: Record<string, string[]> = {
  ember: ['▓▓▓', '▓▓▓', '▓▓▓'],
  openai: ['╭─╮', '│○│', '╰─╯'],
  anthropic: [' ▲ ', '╱ ╲', '───'],
  google: ['╭─╮', '│G│', '╰─╯']
};

export const LogoDisplay: React.FC<LogoDisplayProps> = ({ provider = 'ember', variant = 'full', size = 'medium' }) => {
  const logo = getLogo(provider);
  const useUnicode = supportsUnicode();
  
  if (variant === 'symbol') {
    // Simple symbol display for provider list
    return <Text color={logo.color}>{useUnicode ? logo.symbol : logo.text.charAt(0)}</Text>;
  }

  if (variant === 'ascii' || size === 'large') {
    // Full ASCII art display; fall back to plain ASCII if terminal lacks Unicode
    const asciiLogo = ASCII_LOGOS[provider] || ASCII_LOGOS.ember;
    if (useUnicode) {
      return (
        <Box flexDirection="column" alignItems="center">
          {asciiLogo.map((line, i) => (
            <Text key={i} color={logo.color} bold>{line}</Text>
          ))}
        </Box>
      );
    }
    // Plain ASCII fallback
    const fallback = [
      `[[[ ${logo.text} ]]]`,
    ];
    return (
      <Box flexDirection="column" alignItems="center">
        {fallback.map((line, i) => (
          <Text key={i} color={logo.color} bold>{line}</Text>
        ))}
      </Box>
    );
  }

  if (size === 'small') {
    // Small ASCII display
    const smallLogo = SMALL_LOGOS[provider] || SMALL_LOGOS.ember;
    return (
      <Box flexDirection="column" alignItems="center">
        {smallLogo.map((line, i) => (
          <Text key={i} color={logo.color}>{line}</Text>
        ))}
      </Box>
    );
  }

  // Default medium display
  if (useUnicode) {
    return (
      <Box flexDirection="column" alignItems="center" marginBottom={1}>
        <Text bold color={logo.color}>
          ╔═══════════════════════╗
          ║       {logo.text.padEnd(15, ' ')} ║
          ╚═══════════════════════╝
        </Text>
      </Box>
    );
  }
  // Plain ASCII fallback
  return (
    <Box flexDirection="column" alignItems="center" marginBottom={1}>
      <Text bold color={logo.color}>[ {logo.text} ]</Text>
    </Box>
  );
};
