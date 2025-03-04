import inquirer from 'inquirer';
import inquirerAutocompletePrompt from 'inquirer-autocomplete-prompt';
import inquirerConfirmCommandPrompt from 'inquirer-confirm-command-prompt';
import fuzzy from 'fuzzy';
import chalk from 'chalk';
import { clearLine, clearScreen } from '../utils/terminal';
import { getUIConfig } from './components';

// Register custom inquirer prompts
inquirer.registerPrompt('autocomplete', inquirerAutocompletePrompt);
inquirer.registerPrompt('command', inquirerConfirmCommandPrompt);

/**
 * Options for autocomplete prompts
 */
export interface AutocompleteOptions<T = string> {
  /** Prompt message */
  message: string;
  /** Available choices */
  choices: T[];
  /** Default value */
  default?: T;
  /** Mapper function to extract string representation from choice objects */
  mapChoices?: (choice: T) => string;
  /** Function to sort choices after filtering */
  sort?: (a: T, b: T) => number;
}

/**
 * Options for confirmation prompts
 */
export interface ConfirmOptions {
  /** Prompt message */
  message: string;
  /** Default value */
  default?: boolean;
}

/**
 * Options for input prompts
 */
export interface InputOptions {
  /** Prompt message */
  message: string;
  /** Default value */
  default?: string;
  /** Validation function */
  validate?: (input: string) => boolean | string | Promise<boolean | string>;
  /** Transformation function */
  transform?: (input: string) => string;
}

/**
 * Options for password prompts
 */
export interface PasswordOptions {
  /** Prompt message */
  message: string;
  /** Validation function */
  validate?: (input: string) => boolean | string | Promise<boolean | string>;
}

/**
 * Options for list selection prompts
 */
export interface ListOptions<T = string> {
  /** Prompt message */
  message: string;
  /** Available choices */
  choices: T[];
  /** Default value */
  default?: T;
  /** Number of visible choices */
  pageSize?: number;
  /** Loop the list */
  loop?: boolean;
}

/**
 * Options for checkbox selection prompts
 */
export interface CheckboxOptions<T = string> {
  /** Prompt message */
  message: string;
  /** Available choices */
  choices: T[];
  /** Default value */
  default?: T[];
  /** Number of visible choices */
  pageSize?: number;
  /** Validation function */
  validate?: (selection: T[]) => boolean | string | Promise<boolean | string>;
}

/**
 * Options for command confirmation prompts
 */
export interface CommandOptions {
  /** Prompt message */
  message: string;
  /** Command to confirm */
  command: string;
  /** Default value */
  default?: boolean;
}

/**
 * Interactive prompt utility
 * Handles user interactions with consistent styling
 */
export class Interactive {
  /**
   * Prompts user for confirmation
   * @param options Confirmation options
   * @returns User's boolean choice
   */
  static async confirm(options: ConfirmOptions): Promise<boolean> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or true
    if (quiet || json) {
      return options.default !== undefined ? options.default : true;
    }
    
    const result = await inquirer.prompt<{ value: boolean }>([
      {
        type: 'confirm',
        name: 'value',
        message: options.message,
        default: options.default,
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user for text input
   * @param options Input options
   * @returns User's input
   */
  static async input(options: InputOptions): Promise<string> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or empty string
    if (quiet || json) {
      return options.default || '';
    }
    
    const result = await inquirer.prompt<{ value: string }>([
      {
        type: 'input',
        name: 'value',
        message: options.message,
        default: options.default,
        validate: options.validate,
        transformer: options.transform,
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user for password input
   * @param options Password options
   * @returns User's password
   */
  static async password(options: PasswordOptions): Promise<string> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return empty string
    if (quiet || json) {
      return '';
    }
    
    const result = await inquirer.prompt<{ value: string }>([
      {
        type: 'password',
        name: 'value',
        message: options.message,
        validate: options.validate,
        mask: '*',
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user to select from a list
   * @param options List options
   * @returns User's selection
   */
  static async list<T = string>(options: ListOptions<T>): Promise<T> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or first choice
    if (quiet || json) {
      return options.default !== undefined ? options.default : options.choices[0];
    }
    
    const result = await inquirer.prompt<{ value: T }>([
      {
        type: 'list',
        name: 'value',
        message: options.message,
        choices: options.choices,
        default: options.default,
        pageSize: options.pageSize || 10,
        loop: options.loop !== false,
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user to select multiple items from a list
   * @param options Checkbox options
   * @returns User's selections
   */
  static async checkbox<T = string>(options: CheckboxOptions<T>): Promise<T[]> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or empty array
    if (quiet || json) {
      return options.default || [];
    }
    
    const result = await inquirer.prompt<{ value: T[] }>([
      {
        type: 'checkbox',
        name: 'value',
        message: options.message,
        choices: options.choices,
        default: options.default,
        pageSize: options.pageSize || 10,
        validate: options.validate,
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user with autocomplete
   * @param options Autocomplete options
   * @returns User's selection
   */
  static async autocomplete<T = string>(options: AutocompleteOptions<T>): Promise<T> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or first choice
    if (quiet || json) {
      return options.default !== undefined ? options.default : options.choices[0];
    }
    
    const mapChoices = options.mapChoices || ((choice: T) => String(choice));
    
    const result = await inquirer.prompt<{ value: T }>([
      {
        type: 'autocomplete',
        name: 'value',
        message: options.message,
        default: options.default,
        source: async (answersSoFar: any, input: string = '') => {
          // Filter choices using fuzzy search
          const fuzzyResult = fuzzy.filter(
            input,
            options.choices,
            {
              extract: mapChoices,
            }
          );
          
          // Map back to original choice objects
          let filteredChoices = fuzzyResult.map(result => result.original);
          
          // Sort if a custom sort function is provided
          if (options.sort) {
            filteredChoices = filteredChoices.sort(options.sort);
          }
          
          return filteredChoices;
        },
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Prompts user to confirm a command
   * @param options Command options
   * @returns Whether the command was confirmed
   */
  static async confirmCommand(options: CommandOptions): Promise<boolean> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return default value or true
    if (quiet || json) {
      return options.default !== undefined ? options.default : true;
    }
    
    const result = await inquirer.prompt<{ value: boolean }>([
      {
        type: 'command',
        name: 'value',
        message: options.message,
        command: chalk.bold.cyan(options.command),
        default: options.default,
      },
    ]);
    
    return result.value;
  }
  
  /**
   * Pauses execution until the user presses a key
   * @param message Optional message to display
   */
  static async pressAnyKey(message: string = 'Press any key to continue...'): Promise<void> {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, return immediately
    if (quiet || json) {
      return;
    }
    
    process.stdout.write(message);
    
    return new Promise(resolve => {
      process.stdin.setRawMode(true);
      process.stdin.resume();
      process.stdin.once('data', () => {
        process.stdin.setRawMode(false);
        process.stdin.pause();
        clearLine();
        resolve();
      });
    });
  }
  
  /**
   * Clears the terminal screen
   */
  static clearScreen(): void {
    const { quiet, json } = getUIConfig();
    
    // In non-interactive mode, do nothing
    if (quiet || json) {
      return;
    }
    
    clearScreen();
  }
}

/**
 * Creates a value separator for list choices
 * @returns Inquirer separator
 */
export function createSeparator(): inquirer.Separator {
  return new inquirer.Separator();
}