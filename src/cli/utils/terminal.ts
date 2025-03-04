import { spawn } from 'child_process';
import stripAnsi from 'strip-ansi';
import os from 'os';
import readline from 'readline';

/**
 * Gets the number of columns in the terminal
 * @returns Number of columns in the terminal
 */
export function getTerminalColumns(): number {
  try {
    return process.stdout.columns || 80;
  } catch (e) {
    return 80;
  }
}

/**
 * Gets the number of rows in the terminal
 * @returns Number of rows in the terminal
 */
export function getTerminalRows(): number {
  try {
    return process.stdout.rows || 24;
  } catch (e) {
    return 24;
  }
}

/**
 * Checks if the terminal supports color
 * @returns Whether the terminal supports color
 */
export function supportsColor(): boolean {
  // If NO_COLOR env is set, respect it (https://no-color.org/)
  if ('NO_COLOR' in process.env) {
    return false;
  }

  if (process.env.FORCE_COLOR) {
    return process.env.FORCE_COLOR !== '0';
  }

  if (process.platform === 'win32') {
    // Windows 10 build 14931+ supports 256 color
    const osRelease = os.release().split('.');
    if (
      Number(osRelease[0]) >= 10 &&
      Number(osRelease[2]) >= 14931
    ) {
      return true;
    }
    return false;
  }

  if (process.stdout.isTTY) {
    return true;
  }

  return false;
}

/**
 * Executes a command and streams the output to the console
 * with proper handling for terminal size and colors
 * @param command Command to execute
 * @param args Arguments to pass to the command
 * @param options Options for execution
 * @returns Promise that resolves when the command completes
 */
export function streamCommand(
  command: string,
  args: string[] = [],
  options: {
    /** Environment variables to pass to the command */
    env?: NodeJS.ProcessEnv;
    /** Current working directory */
    cwd?: string;
    /** Whether to stream stderr */
    stderr?: boolean;
    /** Whether to stream stdout */
    stdout?: boolean;
    /** Whether to inherit stdio (takes precedence over stream options) */
    inherit?: boolean;
    /** Callback when data is received from stdout */
    onStdout?: (data: Buffer) => void;
    /** Callback when data is received from stderr */
    onStderr?: (data: Buffer) => void;
  } = {}
): Promise<{ code: number; signal: NodeJS.Signals | null }> {
  const { env = process.env, cwd = process.cwd(), stderr = true, stdout = true, inherit = false } = options;

  return new Promise((resolve, reject) => {
    const childProcess = spawn(command, args, {
      env: { ...env },
      cwd,
      stdio: inherit ? 'inherit' : 'pipe',
    });

    if (!inherit) {
      if (stdout && childProcess.stdout) {
        childProcess.stdout.on('data', (data: Buffer) => {
          if (options.onStdout) {
            options.onStdout(data);
          } else {
            process.stdout.write(data);
          }
        });
      }

      if (stderr && childProcess.stderr) {
        childProcess.stderr.on('data', (data: Buffer) => {
          if (options.onStderr) {
            options.onStderr(data);
          } else {
            process.stderr.write(data);
          }
        });
      }
    }

    childProcess.on('error', (err) => {
      reject(err);
    });

    childProcess.on('close', (code, signal) => {
      resolve({ code: code ?? 0, signal });
    });
  });
}

/**
 * Measures the width of text, taking into account ANSI escape codes
 * @param text Text to measure
 * @returns Width of the text in columns
 */
export function measureTextWidth(text: string): number {
  return stripAnsi(text).length;
}

/**
 * Creates a readline interface for user input
 * @returns Readline interface
 */
export function createReadlineInterface(): readline.Interface {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

/**
 * Clears the current line in the terminal
 */
export function clearLine(): void {
  readline.clearLine(process.stdout, 0);
  readline.cursorTo(process.stdout, 0);
}

/**
 * Moves the cursor up in the terminal
 * @param lines Number of lines to move up
 */
export function moveCursorUp(lines: number): void {
  readline.moveCursor(process.stdout, 0, -lines);
}

/**
 * Clears the terminal screen
 */
export function clearScreen(): void {
  process.stdout.write('\x1b[2J\x1b[0f');
}

/**
 * Detects if the terminal supports Unicode
 * @returns Whether the terminal supports Unicode
 */
export function supportsUnicode(): boolean {
  return process.platform !== 'win32' || 
         Boolean(process.env.CI) || 
         Boolean(process.env.WT_SESSION) || // Windows Terminal
         process.env.TERM_PROGRAM === 'vscode' ||
         process.env.TERM === 'xterm-256color';
}

/**
 * Gets fallback characters for terminals that don't support Unicode
 * @param char Unicode character
 * @returns Fallback character for non-Unicode terminals
 */
export function getFallbackChar(char: string): string {
  if (supportsUnicode()) {
    return char;
  }

  const fallbacks: Record<string, string> = {
    '✓': '√',
    '✖': 'x',
    '⚠': '!',
    'ℹ': 'i',
    '●': 'o',
    '○': 'o',
    '◯': 'o',
    '■': '#',
    '□': '[]',
    '▪': '[',
    '▫': ']',
    '▶': '>',
    '◀': '<',
    '►': '>>',
    '◄': '<<',
    '↑': '^',
    '↓': 'v',
    '←': '<-',
    '→': '->',
    '…': '...',
    '⋮': ':',
    '⌘': 'cmd',
    '⌥': 'alt',
    '⇧': 'shift',
    '⌃': 'ctrl',
  };

  return fallbacks[char] || char;
}