import chalk from 'chalk';
import readline from 'readline';
import figures from 'figures';
import { getTerminalColumns, supportsUnicode, getFallbackChar } from '../utils/terminal';

/**
 * Animated spinner types
 */
export enum SpinnerStyle {
  /** Classic spinner with rotating bar */
  Classic = 'classic',
  /** Dots moving back and forth */
  Dots = 'dots',
  /** Pulse effect with expanding/contracting circle */
  Pulse = 'pulse',
  /** Ember logo pulse effect */
  EmberLogo = 'ember-logo',
  /** Loading text with ellipsis */
  TextEllipsis = 'text-ellipsis',
  /** Minimalist spinner */
  Minimal = 'minimal',
  /** No animation, just text */
  None = 'none'
}

/**
 * Spinner frame sets for different styles
 */
const spinnerFrames: Record<SpinnerStyle, string[]> = {
  [SpinnerStyle.Classic]: ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
  [SpinnerStyle.Dots]: ['⠈', '⠐', '⠠', '⢀', '⡀', '⠄', '⠂', '⠁'],
  [SpinnerStyle.Pulse]: ['○', '◎', '●', '◎'],
  [SpinnerStyle.EmberLogo]: [
    '   ╱╲   ',
    '  ╱  ╲  ',
    ' ╱    ╲ ',
    '╱      ╲',
    '╲      ╱',
    ' ╲    ╱ ',
    '  ╲  ╱  ',
    '   ╲╱   '
  ],
  [SpinnerStyle.TextEllipsis]: [
    'thinking   ',
    'thinking.  ',
    'thinking.. ',
    'thinking...',
    'thinking...',
    'thinking.. ',
    'thinking.  ',
    'thinking   '
  ],
  [SpinnerStyle.Minimal]: ['▪', '▫'],
  [SpinnerStyle.None]: [' ']
};

/**
 * Alternative frames for terminals that don't support Unicode
 */
const fallbackFrames: Record<SpinnerStyle, string[]> = {
  [SpinnerStyle.Classic]: ['|', '/', '-', '\\'],
  [SpinnerStyle.Dots]: ['.  ', '.. ', '...', ' ..', '  .', '   '],
  [SpinnerStyle.Pulse]: ['o', 'O', 'o'],
  [SpinnerStyle.EmberLogo]: [
    '   /\\   ',
    '  /  \\  ',
    ' /    \\ ',
    '/      \\',
    '\\      /',
    ' \\    / ',
    '  \\  /  ',
    '   \\/   '
  ],
  [SpinnerStyle.TextEllipsis]: [
    'thinking   ',
    'thinking.  ',
    'thinking.. ',
    'thinking...',
    'thinking...',
    'thinking.. ',
    'thinking.  ',
    'thinking   '
  ],
  [SpinnerStyle.Minimal]: ['-', ' '],
  [SpinnerStyle.None]: [' ']
};

/**
 * Colored keywords to randomly inject into TextEllipsis spinner
 */
const thinkingKeywords = [
  'analyzing',
  'processing',
  'computing',
  'generating',
  'evaluating',
  'calculating',
  'synthesizing',
  'optimizing',
  'reasoning',
  'inferring',
  'learning',
  'modeling',
  'transforming',
  'embedding',
  'vectorizing',
  'tokenizing',
  'orchestrating',
  'assembling'
];

/**
 * Options for animated spinner
 */
export interface AnimatedSpinnerOptions {
  /** Text to display next to spinner */
  text?: string;
  /** Spinner animation style */
  style?: SpinnerStyle;
  /** Spinner color */
  color?: 'white' | 'cyan' | 'green' | 'yellow' | 'blue' | 'magenta' | 'red' | 'gray';
  /** Text color */
  textColor?: 'white' | 'cyan' | 'green' | 'yellow' | 'blue' | 'magenta' | 'red' | 'gray';
  /** Animation speed in ms */
  interval?: number;
  /** Whether to show succeeded/failed messages */
  showResult?: boolean;
  /** Whether to show timestamps */
  showTimestamp?: boolean;
  /** Whether to use smart text ellipsis with keywords */
  smartEllipsis?: boolean;
}

/**
 * Animated Spinner that supports multiple styles and customizations
 */
export class AnimatedSpinner {
  private text: string;
  private style: SpinnerStyle;
  private color: string;
  private textColor: string;
  private interval: number;
  private frames: string[];
  private frameIndex = 0;
  private isSpinning = false;
  private intervalId: NodeJS.Timeout | null = null;
  private startTime = 0;
  private showResult: boolean;
  private showTimestamp: boolean;
  private smartEllipsis: boolean;
  private lastKeywordTime = 0;
  private currentKeyword = '';
  private lastLineLength = 0;
  
  /**
   * Creates a new animated spinner
   * @param options Spinner options
   */
  constructor(options: AnimatedSpinnerOptions = {}) {
    this.text = options.text || '';
    this.style = options.style || SpinnerStyle.Classic;
    this.color = options.color || 'cyan';
    this.textColor = options.textColor || 'white';
    this.interval = options.interval || 100;
    this.showResult = options.showResult !== false;
    this.showTimestamp = options.showTimestamp || false;
    this.smartEllipsis = options.smartEllipsis || false;
    
    // Use Unicode frames if supported, otherwise fallback frames
    this.frames = supportsUnicode() 
      ? spinnerFrames[this.style] 
      : fallbackFrames[this.style];
  }
  
  /**
   * Starts the spinner animation
   * @param text Optional text to display
   * @returns this instance for chaining
   */
  start(text?: string): AnimatedSpinner {
    if (this.isSpinning) {
      return this;
    }
    
    if (text) {
      this.text = text;
    }
    
    this.startTime = Date.now();
    this.isSpinning = true;
    this.frameIndex = 0;
    this.lastKeywordTime = 0;
    this.currentKeyword = '';
    
    // Start the animation interval
    this.intervalId = setInterval(() => {
      this.render();
    }, this.interval);
    
    // Render immediately
    this.render();
    return this;
  }
  
  /**
   * Updates the spinner text
   * @param text New text to display
   * @returns this instance for chaining
   */
  setText(text: string): AnimatedSpinner {
    this.text = text;
    return this;
  }
  
  /**
   * Changes the spinner style
   * @param style New spinner style
   * @returns this instance for chaining
   */
  setStyle(style: SpinnerStyle): AnimatedSpinner {
    this.style = style;
    this.frames = supportsUnicode() 
      ? spinnerFrames[this.style] 
      : fallbackFrames[this.style];
    return this;
  }
  
  /**
   * Stops the spinner with a success message
   * @param text Optional success message
   * @returns this instance for chaining
   */
  succeed(text?: string): AnimatedSpinner {
    return this.stop(text, 'green', figures.tick);
  }
  
  /**
   * Stops the spinner with an error message
   * @param text Optional error message
   * @returns this instance for chaining
   */
  fail(text?: string): AnimatedSpinner {
    return this.stop(text, 'red', figures.cross);
  }
  
  /**
   * Stops the spinner with a warning message
   * @param text Optional warning message
   * @returns this instance for chaining
   */
  warn(text?: string): AnimatedSpinner {
    return this.stop(text, 'yellow', figures.warning);
  }
  
  /**
   * Stops the spinner with an info message
   * @param text Optional info message
   * @returns this instance for chaining
   */
  info(text?: string): AnimatedSpinner {
    return this.stop(text, 'blue', figures.info);
  }
  
  /**
   * Stops the spinner animation
   * @param text Optional final text
   * @param color Optional color for final text
   * @param symbol Optional symbol to show
   * @returns this instance for chaining
   */
  stop(text?: string, color?: string, symbol?: string): AnimatedSpinner {
    if (!this.isSpinning) {
      return this;
    }
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    
    this.isSpinning = false;
    const finalText = text || this.text;
    
    // Clear the line first
    this.clearLine();
    
    // If showing result, print the final line
    if (this.showResult) {
      const duration = this.getDuration();
      const timestamp = this.showTimestamp ? `[${this.getTimestamp()}] ` : '';
      const symbolStr = symbol ? `${symbol} ` : '';
      
      const finalColor = color || this.color;
      const textColor = (this.textColor === 'white' || !this.textColor) ? 'white' : this.textColor;
      
      const coloredSymbol = symbolStr ? chalk[finalColor](symbolStr) : '';
      const coloredText = chalk[textColor](finalText);
      const finalLine = `${timestamp}${coloredSymbol}${coloredText} ${duration}`;
      
      console.log(finalLine);
    }
    
    return this;
  }
  
  /**
   * Returns the current duration as a formatted string
   * @returns Formatted duration
   */
  private getDuration(): string {
    const duration = Date.now() - this.startTime;
    
    if (duration < 1000) {
      return chalk.dim(`(${duration}ms)`);
    } else if (duration < 60000) {
      return chalk.dim(`(${(duration / 1000).toFixed(1)}s)`);
    } else {
      const minutes = Math.floor(duration / 60000);
      const seconds = ((duration % 60000) / 1000).toFixed(1);
      return chalk.dim(`(${minutes}m ${seconds}s)`);
    }
  }
  
  /**
   * Gets a timestamp string for current time
   * @returns Formatted timestamp
   */
  private getTimestamp(): string {
    const date = new Date();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${hours}:${minutes}:${seconds}`;
  }
  
  /**
   * Renders the current spinner frame
   */
  private render(): void {
    if (!this.isSpinning) {
      return;
    }
    
    const frame = this.frames[this.frameIndex];
    this.frameIndex = (this.frameIndex + 1) % this.frames.length;
    
    const spinner = chalk[this.color](frame);
    let displayText = this.text;
    
    // Smart ellipsis with keywords for thinking spinner
    if (this.smartEllipsis && this.style === SpinnerStyle.TextEllipsis) {
      const now = Date.now();
      
      // Inject a new keyword every 3 seconds
      if (now - this.lastKeywordTime > 3000) {
        this.lastKeywordTime = now;
        const randomKeyword = thinkingKeywords[Math.floor(Math.random() * thinkingKeywords.length)];
        this.currentKeyword = randomKeyword;
      }
      
      if (this.currentKeyword) {
        displayText = displayText.replace('thinking', chalk.magenta(this.currentKeyword));
      }
    }
    
    const textContent = displayText ? chalk[this.textColor](displayText) : '';
    const timestamp = this.showTimestamp ? `[${this.getTimestamp()}] ` : '';
    
    // Special handling for Ember logo spinner
    if (this.style === SpinnerStyle.EmberLogo) {
      this.clearLine();
      const outputLine = `${timestamp}${spinner} ${textContent}`;
      process.stdout.write(outputLine);
      this.lastLineLength = outputLine.length;
    } else {
      this.clearLine();
      const outputLine = `${timestamp}${spinner} ${textContent}`;
      process.stdout.write(outputLine);
      this.lastLineLength = outputLine.length;
    }
  }
  
  /**
   * Gets the current frame without rendering
   * @returns Current spinner frame
   */
  getFrame(): string {
    const frame = this.frames[this.frameIndex];
    return chalk[this.color](frame);
  }
  
  /**
   * Clears the current line in the terminal
   */
  private clearLine(): void {
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
  }
}

/**
 * Creates a new animated spinner with the specified options
 * @param options Spinner options
 * @returns New animated spinner
 */
export function createAnimatedSpinner(options: AnimatedSpinnerOptions = {}): AnimatedSpinner {
  return new AnimatedSpinner(options);
}

/**
 * Creates a thinking spinner with smart keywords
 * @param text Optional initial text
 * @returns New animated spinner
 */
export function createThinkingSpinner(text: string = 'thinking'): AnimatedSpinner {
  return new AnimatedSpinner({
    text,
    style: SpinnerStyle.TextEllipsis,
    color: 'cyan',
    textColor: 'white',
    interval: 120,
    smartEllipsis: true
  });
}

/**
 * Creates an Ember logo spinner
 * @param text Optional text to display
 * @returns New animated spinner
 */
export function createEmberSpinner(text: string = 'Processing'): AnimatedSpinner {
  return new AnimatedSpinner({
    text,
    style: SpinnerStyle.EmberLogo,
    color: 'yellow',
    textColor: 'white',
    interval: 100
  });
}