import chalk from 'chalk';
import readline from 'readline';
import figures from 'figures';
import { getTerminalColumns, clearLine } from '../utils/terminal';
import { formatSize, formatDuration } from './components';
import { AnimatedSpinner, SpinnerStyle } from './animated-spinner';

/**
 * Progress bar styles
 */
export enum ProgressBarStyle {
  /** Standard continuous bar */
  Standard = 'standard',
  /** Blocks style for more granular visuals */
  Blocks = 'blocks',
  /** Simple ASCII style for universal compatibility */
  Simple = 'simple',
  /** Colorful rainbow style */
  Rainbow = 'rainbow',
  /** Customized style for Ember */
  Ember = 'ember'
}

/**
 * Options for animated progress bar
 */
export interface AnimatedProgressOptions {
  /** Total number of steps/items */
  total: number;
  /** Current value */
  current?: number;
  /** Width of the progress bar in characters */
  width?: number;
  /** Progress bar style */
  style?: ProgressBarStyle;
  /** Text to show next to the progress bar */
  text?: string;
  /** Whether to show percentage */
  showPercentage?: boolean;
  /** Whether to show elapsed time */
  showElapsedTime?: boolean;
  /** Whether to show estimated time remaining */
  showEta?: boolean;
  /** Whether to show value/total counts */
  showCount?: boolean;
  /** Whether to show transfer speed (for file operations) */
  showSpeed?: boolean;
  /** Whether to show a spinner next to the progress bar */
  showSpinner?: boolean;
  /** Primary color for the progress bar */
  color?: 'white' | 'cyan' | 'green' | 'yellow' | 'blue' | 'magenta' | 'red' | 'gray';
  /** Whether to clear the progress bar when complete */
  clearOnComplete?: boolean;
  /** How often to update the progress bar (ms) */
  updateInterval?: number;
  /** Auto-start the progress bar */
  autoStart?: boolean;
  /** Whether to handle bytes (for file size formatting) */
  isBytes?: boolean;
  /** Optional custom format string */
  format?: string;
}

/**
 * Progress bar characters for different styles
 */
const progressChars = {
  [ProgressBarStyle.Standard]: {
    complete: '█',
    incomplete: '░',
    head: ''
  },
  [ProgressBarStyle.Blocks]: {
    complete: '█',
    incomplete: '▒',
    head: '▓'
  },
  [ProgressBarStyle.Simple]: {
    complete: '=',
    incomplete: ' ',
    head: '>'
  },
  [ProgressBarStyle.Rainbow]: {
    complete: '█',
    incomplete: '░',
    head: ''
  },
  [ProgressBarStyle.Ember]: {
    complete: '▓',
    incomplete: '░',
    head: '▒'
  }
};

/**
 * Animated progress bar with various styles and features
 */
export class AnimatedProgress {
  private total: number;
  private current: number = 0;
  private width: number;
  private style: ProgressBarStyle;
  private text: string;
  private showPercentage: boolean;
  private showElapsedTime: boolean;
  private showEta: boolean;
  private showCount: boolean;
  private showSpeed: boolean;
  private showSpinner: boolean;
  private color: string;
  private clearOnComplete: boolean;
  private updateInterval: number;
  private isBytes: boolean;
  private format: string | null;

  private startTime: number = 0;
  private endTime: number = 0;
  private lastUpdateTime: number = 0;
  private lastUpdateValue: number = 0;
  private isActive: boolean = false;
  private intervalId: NodeJS.Timeout | null = null;
  private eta: number = 0;
  private speed: number = 0;
  private spinner: AnimatedSpinner | null = null;
  
  /**
   * Creates a new animated progress bar
   * @param options Progress bar options
   */
  constructor(options: AnimatedProgressOptions) {
    this.total = options.total;
    this.current = options.current || 0;
    this.width = options.width || 30;
    this.style = options.style || ProgressBarStyle.Standard;
    this.text = options.text || '';
    this.showPercentage = options.showPercentage !== false;
    this.showElapsedTime = options.showElapsedTime !== false;
    this.showEta = options.showEta !== false;
    this.showCount = options.showCount !== false;
    this.showSpeed = options.showSpeed || false;
    this.showSpinner = options.showSpinner || false;
    this.color = options.color || 'cyan';
    this.clearOnComplete = options.clearOnComplete !== false;
    this.updateInterval = options.updateInterval || 100;
    this.isBytes = options.isBytes || false;
    this.format = options.format || null;
    
    // Initialize spinner if needed
    if (this.showSpinner) {
      this.spinner = new AnimatedSpinner({
        style: SpinnerStyle.Minimal,
        color: this.color,
        showResult: false
      });
    }
    
    // Auto start if specified
    if (options.autoStart) {
      this.start();
    }
  }
  
  /**
   * Starts the progress bar
   * @returns this instance for chaining
   */
  start(): AnimatedProgress {
    if (this.isActive) {
      return this;
    }
    
    this.startTime = Date.now();
    this.lastUpdateTime = this.startTime;
    this.lastUpdateValue = this.current;
    this.isActive = true;
    
    // Start spinner if needed
    if (this.spinner) {
      this.spinner.start();
    }
    
    // Start update interval
    this.intervalId = setInterval(() => {
      this.render();
    }, this.updateInterval);
    
    // Initial render
    this.render();
    
    return this;
  }
  
  /**
   * Updates the progress bar value
   * @param value New value
   * @param text Optional new text
   * @returns this instance for chaining
   */
  update(value: number, text?: string): AnimatedProgress {
    const now = Date.now();
    const timeDelta = now - this.lastUpdateTime;
    const valueDelta = value - this.lastUpdateValue;
    
    // Only calculate speed and ETA if enough time has passed
    if (timeDelta > 100) {
      // Calculate speed (value per second)
      this.speed = (valueDelta / timeDelta) * 1000;
      
      // Calculate ETA (remaining time in ms)
      if (this.speed > 0) {
        this.eta = ((this.total - value) / this.speed) * 1000;
      }
      
      this.lastUpdateTime = now;
      this.lastUpdateValue = value;
    }
    
    this.current = value;
    
    if (text !== undefined) {
      this.text = text;
    }
    
    // If not active, start the progress bar
    if (!this.isActive) {
      this.start();
    } else {
      this.render();
    }
    
    return this;
  }
  
  /**
   * Increments the progress bar by the specified amount
   * @param increment Amount to increment (default: 1)
   * @param text Optional new text
   * @returns this instance for chaining
   */
  increment(increment: number = 1, text?: string): AnimatedProgress {
    return this.update(Math.min(this.current + increment, this.total), text);
  }
  
  /**
   * Sets the total value for the progress bar
   * @param total New total value
   * @returns this instance for chaining
   */
  setTotal(total: number): AnimatedProgress {
    this.total = total;
    return this;
  }
  
  /**
   * Completes the progress bar (sets to 100%)
   * @param text Optional completion text
   * @returns this instance for chaining
   */
  complete(text?: string): AnimatedProgress {
    this.update(this.total, text);
    this.stop();
    return this;
  }
  
  /**
   * Stops the progress bar animation
   * @returns this instance for chaining
   */
  stop(): AnimatedProgress {
    if (!this.isActive) {
      return this;
    }
    
    this.isActive = false;
    this.endTime = Date.now();
    
    // Stop the update interval
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    
    // Stop the spinner
    if (this.spinner) {
      this.spinner.stop();
    }
    
    // Final render
    this.render();
    
    // Add a newline if not clearing
    if (!this.clearOnComplete) {
      process.stdout.write('\n');
    } else {
      clearLine();
    }
    
    return this;
  }
  
  /**
   * Gets the elapsed time in milliseconds
   * @returns Elapsed time
   */
  getElapsedTime(): number {
    if (!this.isActive) {
      return this.endTime - this.startTime;
    }
    return Date.now() - this.startTime;
  }
  
  /**
   * Gets the current progress percentage
   * @returns Progress percentage (0-100)
   */
  getPercentage(): number {
    return Math.min(Math.round((this.current / this.total) * 100), 100);
  }
  
  /**
   * Renders the progress bar
   */
  private render(): void {
    // Format the progress bar
    let output = '';
    const percentage = this.getPercentage();
    
    // Custom format string
    if (this.format) {
      output = this.format
        .replace('{bar}', this.getProgressBar())
        .replace('{percentage}', `${percentage}`)
        .replace('{value}', this.formatValue(this.current))
        .replace('{total}', this.formatValue(this.total))
        .replace('{text}', this.text)
        .replace('{eta}', this.formatEta())
        .replace('{elapsed}', formatDuration(this.getElapsedTime()))
        .replace('{speed}', this.formatSpeed());
    } 
    // Default format
    else {
      // Add spinner if enabled
      if (this.showSpinner && this.spinner) {
        const frame = this.spinner.getFrame();
        output += `${frame} `;
      }
      
      // Add progress bar
      output += this.getProgressBar();
      
      // Add percentage
      if (this.showPercentage) {
        output += ` ${percentage}%`;
      }
      
      // Add count
      if (this.showCount) {
        output += ` ${this.formatValue(this.current)}/${this.formatValue(this.total)}`;
      }
      
      // Add elapsed time
      if (this.showElapsedTime) {
        output += ` ${chalk.dim(formatDuration(this.getElapsedTime()))}`;
      }
      
      // Add ETA
      if (this.showEta && percentage < 100) {
        output += ` ${chalk.dim(`ETA: ${this.formatEta()}`)}`;
      }
      
      // Add speed
      if (this.showSpeed) {
        output += ` ${chalk.dim(this.formatSpeed())}`;
      }
      
      // Add text
      if (this.text) {
        output += ` ${this.text}`;
      }
    }
    
    // Clear the line and write the output
    clearLine();
    process.stdout.write(output);
    
    // If completed and clear on complete, clear the line
    if (percentage >= 100 && this.clearOnComplete && !this.isActive) {
      clearLine();
    }
  }
  
  /**
   * Gets the visual progress bar
   * @returns Formatted progress bar string
   */
  private getProgressBar(): string {
    const chars = progressChars[this.style];
    const availableWidth = this.width;
    const percentage = this.getPercentage();
    
    // Number of complete and incomplete characters
    const completeChars = Math.round((percentage / 100) * availableWidth);
    const incompleteChars = availableWidth - completeChars;
    
    // Build the progress bar
    let bar = '';
    
    // Special handling for rainbow style
    if (this.style === ProgressBarStyle.Rainbow) {
      const rainbowColors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta'];
      
      // Handle the completed part with rainbow colors
      for (let i = 0; i < completeChars; i++) {
        const colorIndex = i % rainbowColors.length;
        bar += chalk[rainbowColors[colorIndex]](chars.complete);
      }
      
      // Handle the incomplete part
      bar += chalk.dim(chars.incomplete.repeat(incompleteChars));
    } 
    // Normal handling for other styles
    else {
      // Complete part
      bar += chalk[this.color](chars.complete.repeat(completeChars));
      
      // Head character (if defined and not at 100%)
      if (chars.head && completeChars < availableWidth) {
        bar += chalk[this.color](chars.head);
        // Adjust the incomplete count
        if (incompleteChars > 0) {
          bar += chalk.dim(chars.incomplete.repeat(incompleteChars - 1));
        }
      } else {
        // No head character, just the incomplete part
        bar += chalk.dim(chars.incomplete.repeat(incompleteChars));
      }
    }
    
    return `[${bar}]`;
  }
  
  /**
   * Formats the ETA value
   * @returns Formatted ETA string
   */
  private formatEta(): string {
    if (this.eta <= 0 || !isFinite(this.eta)) {
      return '--';
    }
    return formatDuration(this.eta);
  }
  
  /**
   * Formats the speed value
   * @returns Formatted speed string
   */
  private formatSpeed(): string {
    if (this.speed <= 0 || !isFinite(this.speed)) {
      return '--/s';
    }
    
    if (this.isBytes) {
      return `${formatSize(this.speed)}/s`;
    }
    
    // Format to 1 decimal place if small, otherwise round
    if (this.speed < 10) {
      return `${this.speed.toFixed(1)}/s`;
    }
    return `${Math.round(this.speed)}/s`;
  }
  
  /**
   * Formats a value (with byte handling if appropriate)
   * @param value Value to format
   * @returns Formatted value string
   */
  private formatValue(value: number): string {
    if (this.isBytes) {
      return formatSize(value);
    }
    return value.toString();
  }
}

/**
 * Creates a new animated progress bar
 * @param options Progress bar options
 * @returns New animated progress bar
 */
export function createProgress(options: AnimatedProgressOptions): AnimatedProgress {
  return new AnimatedProgress(options);
}

/**
 * Creates a file download progress bar
 * @param total Total file size in bytes
 * @param text Optional description text
 * @returns New animated progress bar configured for downloads
 */
export function createDownloadProgress(total: number, text: string = 'Downloading'): AnimatedProgress {
  return new AnimatedProgress({
    total,
    text,
    style: ProgressBarStyle.Blocks,
    color: 'cyan',
    showSpeed: true,
    isBytes: true,
    autoStart: true
  });
}

/**
 * Creates an Ember-styled progress bar
 * @param total Total number of steps
 * @param text Optional description text
 * @returns New animated progress bar with Ember styling
 */
export function createEmberProgress(total: number, text: string = ''): AnimatedProgress {
  return new AnimatedProgress({
    total,
    text,
    style: ProgressBarStyle.Ember,
    color: 'yellow',
    showSpinner: true,
    autoStart: true
  });
}