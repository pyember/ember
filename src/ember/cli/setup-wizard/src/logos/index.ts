/**
 * Minimalist, professional logo system.
 * Inspired by Swiss design principles - clean, functional, beautiful.
 */

export interface Logo {
  symbol: string;     // Single character representation
  text: string;       // Clean text representation
  color: string;      // Brand color
}

export const LOGOS: Record<string, Logo> = {
  ember: {
    symbol: '◆',      // Diamond - represents precision and value
    text: 'EMBER',
    color: 'blue'
  },
  openai: {
    symbol: '○',      // Circle - represents completeness and AI
    text: 'OpenAI',
    color: 'green'
  },
  anthropic: {
    symbol: '△',      // Triangle - represents stability and advancement  
    text: 'Anthropic',
    color: 'yellow'
  },
  google: {
    symbol: '◎',      // Double circle - represents search and discovery
    text: 'Google',
    color: 'blue'
  }
};

/**
 * Get minimalist logo representation.
 */
export function getLogo(provider: string): Logo {
  return LOGOS[provider] || {
    symbol: provider.charAt(0).toUpperCase(),
    text: provider,
    color: 'white'
  };
}

/**
 * Check if terminal supports Unicode.
 */
export function supportsUnicode(): boolean {
  const env = process.env;
  return (
    env.TERM !== 'linux' &&
    (env.LANG?.includes('UTF-8') ||
    env.LC_ALL?.includes('UTF-8') ||
    env.LC_CTYPE?.includes('UTF-8')) || false
  );
}

/**
 * Format logo for inline display with consistent spacing.
 */
export function formatLogoInline(provider: string): string {
  const logo = getLogo(provider);
  return logo.symbol;
}