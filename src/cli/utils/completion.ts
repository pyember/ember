/**
 * Shell Completion Generator for Ember CLI
 * 
 * This module generates shell completion scripts for Bash, Zsh, Fish, and PowerShell.
 * It uses a declarative approach to define completions and delegates the generation
 * of shell-specific code to specialized generators.
 * 
 * Design principles:
 * - Single Responsibility: Each generator handles one shell
 * - Open/Closed: New shells can be added without modifying existing code
 * - Liskov Substitution: All generators share a common interface
 * - Interface Segregation: Clean, focused interfaces
 * - Dependency Inversion: High-level modules don't depend on low-level details
 */

import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { getPythonBridge } from '../bridge/python-bridge';
import { ConfigManager } from '../services/config-manager';

/**
 * Command Completion Definition
 * Represents the structure of a command for autocompletion
 */
interface CompletionCommand {
  /** Command name */
  name: string;
  
  /** Command description */
  description: string;
  
  /** Command options */
  options?: CompletionOption[];
  
  /** Command arguments */
  args?: CompletionArgument[];
  
  /** Subcommands */
  subcommands?: CompletionCommand[];
}

/**
 * Command Option Definition
 * Represents a command-line option for autocompletion
 */
interface CompletionOption {
  /** Option name (without dashes) */
  name: string;
  
  /** Short form (single letter, without dash) */
  short?: string;
  
  /** Option description */
  description: string;
  
  /** Whether the option takes a value */
  takesValue?: boolean;
  
  /** 
   * Value completion type 
   * Can be a primitive type or a completion source function name
   */
  valueType?: 'file' | 'directory' | 'provider' | 'model' | 'template' | string;
}

/**
 * Command Argument Definition
 * Represents a positional argument for autocompletion
 */
interface CompletionArgument {
  /** Argument name */
  name: string;
  
  /** Argument description */
  description: string;
  
  /** Whether the argument is required */
  required?: boolean;
  
  /** 
   * Argument completion type
   * Can be a primitive type or a completion source function name
   */
  completionType?: 'file' | 'directory' | 'provider' | 'model' | 'template' | string;
}

/**
 * Shell Completion Generator Interface
 * Abstract interface for shell-specific completion generators
 */
interface ShellCompletionGenerator {
  /**
   * Generate completion script for a specific shell
   * 
   * @param commands Command tree for completion
   * @param programName Program name (default: ember)
   * @returns Completion script as a string
   */
  generate(commands: CompletionCommand[], programName?: string): string;
}

/**
 * Bash Completion Generator
 * Generates Bash completion scripts using compgen and complete
 */
class BashCompletionGenerator implements ShellCompletionGenerator {
  /**
   * Generate Bash completion script
   * 
   * @param commands Command tree
   * @param programName Program name (default: ember)
   * @returns Bash completion script
   */
  generate(commands: CompletionCommand[], programName: string = 'ember'): string {
    const scriptParts: string[] = [
      `#!/usr/bin/env bash`,
      ``,
      `# Ember CLI Bash completion script`,
      `# Generated automatically, do not modify`,
      ``,
      `_${programName}_completions() {`,
      `  local cur prev opts`,
      `  COMPREPLY=()`,
      `  cur="${'${COMP_WORDS[COMP_CWORD]}'}"}`,
      `  prev="${'${COMP_WORDS[COMP_CWORD-1]}'}"}`,
      `  cmd="${'${COMP_WORDS[1]}'}"}`,
      `  subcmd="${'${COMP_WORDS[2]}'}"}`,
      ``
    ];
    
    // Generate completion functions for dynamic sources
    scriptParts.push(this.generateDynamicCompletionFunctions(programName));
    
    // Main command switch
    scriptParts.push(`  # Main command switch`);
    scriptParts.push(`  case "$cmd" in`);
    
    // Generate case statements for top-level commands
    for (const cmd of commands) {
      scriptParts.push(`    ${cmd.name})`);
      
      if (cmd.subcommands && cmd.subcommands.length > 0) {
        // If it has subcommands, handle them
        scriptParts.push(`      case "$subcmd" in`);
        
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`        ${subcmd.name})`);
          
          // Handle options for this subcommand
          if (subcmd.options && subcmd.options.length > 0) {
            const optionLines = this.generateOptionCompletions(subcmd.options);
            scriptParts.push(...optionLines.map(line => `          ${line}`));
          }
          
          // Handle arguments for this subcommand
          if (subcmd.args && subcmd.args.length > 0) {
            const argLines = this.generateArgumentCompletions(subcmd.args);
            scriptParts.push(...argLines.map(line => `          ${line}`));
          }
          
          scriptParts.push(`          ;;`);
        }
        
        // Add subcommand completion when no subcommand is provided yet
        scriptParts.push(`        *)`);
        const subcommandNames = cmd.subcommands.map(subcmd => subcmd.name).join(' ');
        scriptParts.push(`          COMPREPLY=($(compgen -W "${subcommandNames}" -- "$cur"))`);
        scriptParts.push(`          ;;`);
        scriptParts.push(`      esac`);
      } else {
        // If it's a simple command, just complete options
        if (cmd.options && cmd.options.length > 0) {
          const optionLines = this.generateOptionCompletions(cmd.options);
          scriptParts.push(...optionLines.map(line => `      ${line}`));
        }
        
        // Handle arguments for this command
        if (cmd.args && cmd.args.length > 0) {
          const argLines = this.generateArgumentCompletions(cmd.args);
          scriptParts.push(...argLines.map(line => `      ${line}`));
        }
      }
      
      scriptParts.push(`      ;;`);
    }
    
    // Default case: complete top-level commands
    scriptParts.push(`    *)`);
    const commandNames = commands.map(cmd => cmd.name).join(' ');
    scriptParts.push(`      COMPREPLY=($(compgen -W "${commandNames}" -- "$cur"))`);
    scriptParts.push(`      ;;`);
    scriptParts.push(`  esac`);
    scriptParts.push(`}`);
    scriptParts.push(``);
    
    // Register the completion function
    scriptParts.push(`complete -F _${programName}_completions ${programName}`);
    scriptParts.push(``);
    
    return scriptParts.join('\n');
  }
  
  /**
   * Generate option completion code for Bash
   * 
   * @param options List of command options
   * @returns Array of code lines for option completion
   */
  private generateOptionCompletions(options: CompletionOption[]): string[] {
    const lines: string[] = [];
    
    // Generate list of options for completion
    const optionsList = options.map(opt => 
      opt.short ? `--${opt.name} -${opt.short}` : `--${opt.name}`
    ).join(' ');
    
    lines.push(`if [[ "$cur" == -* ]]; then`);
    lines.push(`  COMPREPLY=($(compgen -W "${optionsList}" -- "$cur"))`);
    lines.push(`  return`);
    lines.push(`fi`);
    
    // Handle options that take values
    lines.push(`case "$prev" in`);
    
    for (const opt of options) {
      if (opt.takesValue && opt.valueType) {
        const optNames = [
          `--${opt.name}`,
          ...(opt.short ? [`-${opt.short}`] : [])
        ].join('|');
        
        lines.push(`  ${optNames})`);
        
        // Handle different value types
        switch (opt.valueType) {
          case 'file':
            lines.push(`    COMPREPLY=($(compgen -f -- "$cur"))`);
            break;
          case 'directory':
            lines.push(`    COMPREPLY=($(compgen -d -- "$cur"))`);
            break;
          case 'provider':
            lines.push(`    COMPREPLY=($(${programName}_complete_providers "$cur"))`);
            break;
          case 'model':
            lines.push(`    COMPREPLY=($(${programName}_complete_models "$cur"))`);
            break;
          case 'template':
            lines.push(`    COMPREPLY=($(${programName}_complete_templates "$cur"))`);
            break;
          default:
            // If it's a custom completion function
            if (typeof opt.valueType === 'string') {
              lines.push(`    COMPREPLY=($(${opt.valueType} "$cur"))`);
            }
            break;
        }
        
        lines.push(`    return`);
        lines.push(`    ;;`);
      }
    }
    
    lines.push(`esac`);
    
    return lines;
  }
  
  /**
   * Generate argument completion code for Bash
   * 
   * @param args List of command arguments
   * @returns Array of code lines for argument completion
   */
  private generateArgumentCompletions(args: CompletionArgument[]): string[] {
    const lines: string[] = [];
    
    // We need to determine which argument we're on
    lines.push(`# Calculate which positional argument we're completing`);
    lines.push(`local argument_position=0`);
    lines.push(`for ((i=1; i < COMP_CWORD; i++)); do`);
    lines.push(`  if [[ "${'${COMP_WORDS[i]}'}}" != -* ]]; then`);
    lines.push(`    ((argument_position++))`);
    lines.push(`  fi`);
    lines.push(`done`);
    
    // Handle each possible argument position
    lines.push(`case "$argument_position" in`);
    
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      lines.push(`  ${i})`);
      
      // Handle different completion types
      if (arg.completionType) {
        switch (arg.completionType) {
          case 'file':
            lines.push(`    COMPREPLY=($(compgen -f -- "$cur"))`);
            break;
          case 'directory':
            lines.push(`    COMPREPLY=($(compgen -d -- "$cur"))`);
            break;
          case 'provider':
            lines.push(`    COMPREPLY=($(${programName}_complete_providers "$cur"))`);
            break;
          case 'model':
            lines.push(`    COMPREPLY=($(${programName}_complete_models "$cur"))`);
            break;
          case 'template':
            lines.push(`    COMPREPLY=($(${programName}_complete_templates "$cur"))`);
            break;
          default:
            // If it's a custom completion function
            if (typeof arg.completionType === 'string') {
              lines.push(`    COMPREPLY=($(${arg.completionType} "$cur"))`);
            }
            break;
        }
      }
      
      lines.push(`    ;;`);
    }
    
    lines.push(`esac`);
    
    return lines;
  }
  
  /**
   * Generate functions for dynamic completion sources
   * 
   * @param programName Program name
   * @returns Bash functions for dynamic completions
   */
  private generateDynamicCompletionFunctions(programName: string): string {
    return `
  # Dynamic completion functions
  ${programName}_complete_providers() {
    local cur=$1
    local providers=$(${programName} provider list --json 2>/dev/null | grep -o '"providers":\\[[^]]*\\]' | grep -o '"[^"]*"' | tr -d '"')
    COMPREPLY=($(compgen -W "$providers" -- "$cur"))
    return 0
  }

  ${programName}_complete_models() {
    local cur=$1
    local models=$(${programName} model list --json 2>/dev/null | grep -o '"models":\\[[^]]*\\]' | grep -o '"[^"]*"' | tr -d '"')
    COMPREPLY=($(compgen -W "$models" -- "$cur"))
    return 0
  }

  ${programName}_complete_templates() {
    local cur=$1
    local templates=$(${programName} project templates --json 2>/dev/null | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    COMPREPLY=($(compgen -W "$templates" -- "$cur"))
    return 0
  }
`;
  }
}

/**
 * Zsh Completion Generator
 * Generates Zsh completion scripts using _arguments and _describe
 */
class ZshCompletionGenerator implements ShellCompletionGenerator {
  /**
   * Generate Zsh completion script
   * 
   * @param commands Command tree
   * @param programName Program name (default: ember)
   * @returns Zsh completion script
   */
  generate(commands: CompletionCommand[], programName: string = 'ember'): string {
    const scriptParts: string[] = [
      `#compdef ${programName}`,
      ``,
      `# Ember CLI Zsh completion script`,
      `# Generated automatically, do not modify`,
      ``,
      `_${programName}() {`,
      `  local -a commands`,
      `  local -a subcommands`,
      `  local curcontext="$curcontext" state line`,
      `  typeset -A opt_args`,
      ``
    ];
    
    // Generate functions for dynamic sources
    scriptParts.push(this.generateDynamicCompletionFunctions(programName));
    
    // Define top-level commands
    scriptParts.push(`  # Top-level commands`);
    scriptParts.push(`  commands=(`);
    for (const cmd of commands) {
      scriptParts.push(`    "${cmd.name}:${cmd.description}"`);
    }
    scriptParts.push(`  )`);
    scriptParts.push(``);
    
    // Main command switch
    scriptParts.push(`  _arguments -C \\`);
    scriptParts.push(`    "1: :{_describe 'commands' commands}" \\`);
    scriptParts.push(`    "*::arg:->args" && return`);
    scriptParts.push(``);
    scriptParts.push(`  case $line[1] in`);
    
    // Generate case statements for top-level commands
    for (const cmd of commands) {
      scriptParts.push(`    ${cmd.name})`);
      
      if (cmd.subcommands && cmd.subcommands.length > 0) {
        // Define subcommands for this command
        scriptParts.push(`      subcommands=(`);
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`        "${subcmd.name}:${subcmd.description}"`);
        }
        scriptParts.push(`      )`);
        
        // Subcommand handling
        scriptParts.push(`      if (( CURRENT == 1 )); then`);
        scriptParts.push(`        _describe "${cmd.name} subcommands" subcommands && ret=0`);
        scriptParts.push(`        return`);
        scriptParts.push(`      fi`);
        scriptParts.push(`      case $line[1] in`);
        
        // Handle each subcommand
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`        ${subcmd.name})`);
          
          // Generate _arguments line for this subcommand
          const argLines = this.generateZshArguments(subcmd.options || [], subcmd.args || []);
          scriptParts.push(...argLines.map(line => `          ${line}`));
          
          scriptParts.push(`          ;;`);
        }
        
        scriptParts.push(`      esac`);
      } else {
        // Generate _arguments line for simple command
        const argLines = this.generateZshArguments(cmd.options || [], cmd.args || []);
        scriptParts.push(...argLines.map(line => `      ${line}`));
      }
      
      scriptParts.push(`      ;;`);
    }
    
    scriptParts.push(`  esac`);
    scriptParts.push(`}`);
    scriptParts.push(``);
    scriptParts.push(`_${programName}`);
    scriptParts.push(``);
    
    return scriptParts.join('\n');
  }
  
  /**
   * Generate Zsh _arguments specification for a command
   * 
   * @param options Command options
   * @param args Command arguments
   * @returns Array of code lines for _arguments
   */
  private generateZshArguments(options: CompletionOption[], args: CompletionArgument[]): string[] {
    const lines: string[] = [];
    const argSpecs: string[] = [];
    
    // Generate option specs
    for (const opt of options) {
      let spec = '';
      
      // Generate option flag specs
      if (opt.short) {
        spec += `{-${opt.short},--${opt.name}}`;
      } else {
        spec += `--${opt.name}`;
      }
      
      // Add value handling if needed
      if (opt.takesValue && opt.valueType) {
        let completionSpec: string;
        
        // Handle different value types
        switch (opt.valueType) {
          case 'file':
            completionSpec = '_files';
            break;
          case 'directory':
            completionSpec = '_directories';
            break;
          case 'provider':
            completionSpec = '_ember_providers';
            break;
          case 'model':
            completionSpec = '_ember_models';
            break;
          case 'template':
            completionSpec = '_ember_templates';
            break;
          default:
            // Custom completion
            completionSpec = opt.valueType;
            break;
        }
        
        spec += `+"[${opt.description}]":${opt.name}:"${completionSpec}"`;
      } else {
        spec += `"[${opt.description}]"`;
      }
      
      argSpecs.push(spec);
    }
    
    // Generate argument specs
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      let spec = '';
      
      // Required or optional argument
      if (arg.required) {
        spec += `${i + 1}:${arg.description}:`;
      } else {
        spec += `${i + 1}::${arg.description}:`;
      }
      
      // Add completion if specified
      if (arg.completionType) {
        let completionSpec: string;
        
        // Handle different value types
        switch (arg.completionType) {
          case 'file':
            completionSpec = '_files';
            break;
          case 'directory':
            completionSpec = '_directories';
            break;
          case 'provider':
            completionSpec = '_ember_providers';
            break;
          case 'model':
            completionSpec = '_ember_models';
            break;
          case 'template':
            completionSpec = '_ember_templates';
            break;
          default:
            // Custom completion
            completionSpec = arg.completionType;
            break;
        }
        
        spec += completionSpec;
      }
      
      argSpecs.push(spec);
    }
    
    // Format the _arguments spec
    if (argSpecs.length > 0) {
      lines.push(`_arguments \\`);
      for (let i = 0; i < argSpecs.length; i++) {
        const isLast = i === argSpecs.length - 1;
        lines.push(`  "${argSpecs[i]}"${isLast ? '' : ' \\'}`);
      }
    } else {
      lines.push(`_arguments`);
    }
    
    return lines;
  }
  
  /**
   * Generate functions for dynamic completion sources in Zsh
   * 
   * @param programName Program name
   * @returns Zsh functions for dynamic completions
   */
  private generateDynamicCompletionFunctions(programName: string): string {
    return `  # Dynamic completion functions
  _ember_providers() {
    local -a providers
    providers=($(${programName} provider list --json 2>/dev/null | grep -o '"providers":\\[[^]]*\\]' | grep -o '"[^"]*"' | tr -d '"'))
    _describe 'providers' providers
  }

  _ember_models() {
    local -a models
    models=($(${programName} model list --json 2>/dev/null | grep -o '"models":\\[[^]]*\\]' | grep -o '"[^"]*"' | tr -d '"'))
    _describe 'models' models
  }

  _ember_templates() {
    local -a templates
    templates=($(${programName} project templates --json 2>/dev/null | grep -o '"id":"[^"]*"' | cut -d'"' -f4))
    _describe 'templates' templates
  }
`;
  }
}

/**
 * Fish Completion Generator
 * Generates Fish completion scripts using the complete command
 */
class FishCompletionGenerator implements ShellCompletionGenerator {
  /**
   * Generate Fish completion script
   * 
   * @param commands Command tree
   * @param programName Program name (default: ember)
   * @returns Fish completion script
   */
  generate(commands: CompletionCommand[], programName: string = 'ember'): string {
    const scriptParts: string[] = [
      `# Ember CLI Fish completion script`,
      `# Generated automatically, do not modify`,
      ``
    ];
    
    // Add functions for dynamic completions
    scriptParts.push(this.generateDynamicCompletionFunctions(programName));
    scriptParts.push(``);
    
    // Clear any existing completions for the program
    scriptParts.push(`# Clear existing completions`);
    scriptParts.push(`complete -c ${programName} -e`);
    scriptParts.push(``);
    
    // Add top-level command completions
    scriptParts.push(`# Top-level commands`);
    for (const cmd of commands) {
      scriptParts.push(`complete -c ${programName} -f -n "__fish_use_subcommand" -a "${cmd.name}" -d "${cmd.description}"`);
    }
    scriptParts.push(``);
    
    // Add subcommand and options completions
    for (const cmd of commands) {
      // If the command has subcommands
      if (cmd.subcommands && cmd.subcommands.length > 0) {
        scriptParts.push(`# Subcommands for ${cmd.name}`);
        
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`complete -c ${programName} -f -n "__fish_seen_subcommand_from ${cmd.name}; and not __fish_seen_subcommand_from ${subcmd.subcommands?.map(sc => sc.name).join(' ') || ''}" -a "${subcmd.name}" -d "${subcmd.description}"`);
          
          // Add options for this subcommand
          if (subcmd.options && subcmd.options.length > 0) {
            for (const opt of subcmd.options) {
              const shortFlag = opt.short ? `-s ${opt.short}` : '';
              const valueSpec = opt.takesValue ? ` -r ${this.getCompletionForType(opt.valueType, programName)}` : '';
              
              scriptParts.push(`complete -c ${programName} -f -n "__fish_seen_subcommand_from ${cmd.name}; and __fish_seen_subcommand_from ${subcmd.name}" -l "${opt.name}" ${shortFlag} -d "${opt.description}"${valueSpec}`);
            }
          }
          
          // Add args for this subcommand
          if (subcmd.args && subcmd.args.length > 0) {
            for (let i = 0; i < subcmd.args.length; i++) {
              const arg = subcmd.args[i];
              const position = i + 1;
              const valueSpec = arg.completionType ? ` ${this.getCompletionForType(arg.completionType, programName)}` : '';
              
              scriptParts.push(`complete -c ${programName} -f -n "__fish_seen_subcommand_from ${cmd.name}; and __fish_seen_subcommand_from ${subcmd.name}; and __fish_is_nth_token ${position + 2}" -d "${arg.description}"${valueSpec}`);
            }
          }
        }
      } else {
        // Add options for this command
        if (cmd.options && cmd.options.length > 0) {
          scriptParts.push(`# Options for ${cmd.name}`);
          
          for (const opt of cmd.options) {
            const shortFlag = opt.short ? `-s ${opt.short}` : '';
            const valueSpec = opt.takesValue ? ` -r ${this.getCompletionForType(opt.valueType, programName)}` : '';
            
            scriptParts.push(`complete -c ${programName} -f -n "__fish_seen_subcommand_from ${cmd.name}" -l "${opt.name}" ${shortFlag} -d "${opt.description}"${valueSpec}`);
          }
        }
        
        // Add args for this command
        if (cmd.args && cmd.args.length > 0) {
          scriptParts.push(`# Arguments for ${cmd.name}`);
          
          for (let i = 0; i < cmd.args.length; i++) {
            const arg = cmd.args[i];
            const position = i + 1;
            const valueSpec = arg.completionType ? ` ${this.getCompletionForType(arg.completionType, programName)}` : '';
            
            scriptParts.push(`complete -c ${programName} -f -n "__fish_seen_subcommand_from ${cmd.name}; and __fish_is_nth_token ${position + 1}" -d "${arg.description}"${valueSpec}`);
          }
        }
      }
      
      scriptParts.push(``);
    }
    
    // Add global options
    scriptParts.push(`# Global options`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "debug" -d "Enable debug mode"`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "json" -d "Output as JSON"`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "quiet" -d "Suppress non-essential output"`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "no-color" -d "Disable colors"`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "version" -s v -d "Display version information"`);
    scriptParts.push(`complete -c ${programName} -f -n "true" -l "help" -s h -d "Display help information"`);
    
    return scriptParts.join('\n');
  }
  
  /**
   * Get Fish completion specification for a value type
   * 
   * @param valueType Type of value to complete
   * @param programName Program name for dynamic completions
   * @returns Fish completion specification
   */
  private getCompletionForType(valueType?: string, programName: string = 'ember'): string {
    if (!valueType) return '';
    
    switch (valueType) {
      case 'file':
        return '-a "(__fish_complete_path)"';
      case 'directory':
        return '-a "(__fish_complete_directories)"';
      case 'provider':
        return `-a "(__${programName}_complete_providers)"`;
      case 'model':
        return `-a "(__${programName}_complete_models)"`;
      case 'template':
        return `-a "(__${programName}_complete_templates)"`;
      default:
        // Custom completion
        return valueType.startsWith('-a ') ? valueType : `-a "${valueType}"`;
    }
  }
  
  /**
   * Generate functions for dynamic completion sources in Fish
   * 
   * @param programName Program name
   * @returns Fish functions for dynamic completions
   */
  private generateDynamicCompletionFunctions(programName: string): string {
    return `# Dynamic completion functions
function __${programName}_complete_providers
    ${programName} provider list --json 2>/dev/null | string match -r '"providers":\\[[^]]*\\]' | string match -r '"[^"]*"' | string replace -a '"' ''
end

function __${programName}_complete_models
    ${programName} model list --json 2>/dev/null | string match -r '"models":\\[[^]]*\\]' | string match -r '"[^"]*"' | string replace -a '"' ''
end

function __${programName}_complete_templates
    ${programName} project templates --json 2>/dev/null | string match -r '"id":"[^"]*"' | string replace -r '.*"id":"([^"]*)".*' '$1'
end`;
  }
}

/**
 * PowerShell Completion Generator
 * Generates PowerShell completion using Register-ArgumentCompleter
 */
class PowerShellCompletionGenerator implements ShellCompletionGenerator {
  /**
   * Generate PowerShell completion script
   * 
   * @param commands Command tree
   * @param programName Program name (default: ember)
   * @returns PowerShell completion script
   */
  generate(commands: CompletionCommand[], programName: string = 'ember'): string {
    const scriptParts: string[] = [
      `# Ember CLI PowerShell completion script`,
      `# Generated automatically, do not modify`,
      ``,
      `# Register argument completer for ${programName}`,
      `Register-ArgumentCompleter -Native -CommandName ${programName} -ScriptBlock {`,
      `    param($wordToComplete, $commandAst, $cursorPosition)`,
      ``,
      `    # Extract information from command line`,
      `    $commandElements = $commandAst.CommandElements`,
      `    $command = @(`,
      `        $commandElements |`,
      `            Where-Object { $_ -is [System.Management.Automation.Language.StringConstantExpressionAst] } |`,
      `            Select-Object -ExpandProperty Value`,
      `    )`,
      `    $commandLine = $commandAst.Extent.Text`,
      ``,
      `    # Helper function to get dynamic completions`,
      `    function Get-EmberDynamicCompletions {`,
      `        param ([string]$type, [string]$filter = "")`,
      `        switch ($type) {`,
      `            "provider" {`,
      `                $json = & ${programName} provider list --json 2>$null`,
      `                if ($json) {`,
      `                    $providers = ($json | ConvertFrom-Json).providers`,
      `                    $providers | Where-Object { $_ -like "$filter*" }`,
      `                }`,
      `            }`,
      `            "model" {`,
      `                $json = & ${programName} model list --json 2>$null`,
      `                if ($json) {`,
      `                    $models = ($json | ConvertFrom-Json).models`,
      `                    $models | Where-Object { $_ -like "$filter*" }`,
      `                }`,
      `            }`,
      `            "template" {`,
      `                $json = & ${programName} project templates --json 2>$null`,
      `                if ($json) {`,
      `                    $templates = ($json | ConvertFrom-Json).templates | ForEach-Object { $_.id }`,
      `                    $templates | Where-Object { $_ -like "$filter*" }`,
      `                }`,
      `            }`,
      `        }`,
      `    }`,
      ``
    ];
    
    // Top-level command handling
    scriptParts.push(`    # If only ember is typed, complete with top-level commands`);
    scriptParts.push(`    if ($commandElements.Count -eq 1) {`);
    scriptParts.push(`        @(`);
    for (const cmd of commands) {
      scriptParts.push(`            [System.Management.Automation.CompletionResult]::new('${cmd.name}', '${cmd.name}', 'ParameterValue', '${cmd.description.replace(/'/g, "''")}')`);
    }
    scriptParts.push(`        ) | Where-Object { $_.CompletionText -like "$wordToComplete*" }`);
    scriptParts.push(`        return`);
    scriptParts.push(`    }`);
    scriptParts.push(``);
    
    // Command-specific handling
    scriptParts.push(`    # Command-specific completion`);
    scriptParts.push(`    switch ($command[1]) {`);
    
    for (const cmd of commands) {
      scriptParts.push(`        "${cmd.name}" {`);
      
      // If the command has subcommands
      if (cmd.subcommands && cmd.subcommands.length > 0) {
        scriptParts.push(`            # Subcommand handling for ${cmd.name}`);
        scriptParts.push(`            if ($commandElements.Count -eq 2) {`);
        scriptParts.push(`                # If only the command is typed, complete with subcommands`);
        scriptParts.push(`                @(`);
        
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`                    [System.Management.Automation.CompletionResult]::new('${subcmd.name}', '${subcmd.name}', 'ParameterValue', '${subcmd.description.replace(/'/g, "''")}')`);
        }
        
        scriptParts.push(`                ) | Where-Object { $_.CompletionText -like "$wordToComplete*" }`);
        scriptParts.push(`                return`);
        scriptParts.push(`            }`);
        scriptParts.push(``);
        
        // Handle subcommands
        scriptParts.push(`            # Specific subcommand handling`);
        scriptParts.push(`            switch ($command[2]) {`);
        
        for (const subcmd of cmd.subcommands) {
          scriptParts.push(`                "${subcmd.name}" {`);
          
          // Handle options for this subcommand
          if (subcmd.options && subcmd.options.length > 0) {
            scriptParts.push(`                    # Handle options for ${subcmd.name}`);
            scriptParts.push(`                    if ($wordToComplete -like "-*") {`);
            scriptParts.push(`                        @(`);
            
            for (const opt of subcmd.options) {
              const shortForm = opt.short ? `, '-${opt.short}'` : '';
              scriptParts.push(`                            [System.Management.Automation.CompletionResult]::new('--${opt.name}', '--${opt.name}'${shortForm}, 'ParameterValue', '${opt.description.replace(/'/g, "''")}')`);
            }
            
            scriptParts.push(`                        ) | Where-Object { $_.CompletionText -like "$wordToComplete*" }`);
            scriptParts.push(`                        return`);
            scriptParts.push(`                    }`);
            scriptParts.push(``);
            
            // Handle option arguments
            scriptParts.push(`                    # Handle option arguments`);
            scriptParts.push(`                    $previousParameter = $commandElements[$commandElements.Count - 2].Extent.Text`);
            scriptParts.push(`                    switch ($previousParameter) {`);
            
            for (const opt of subcmd.options) {
              if (opt.takesValue && opt.valueType) {
                const optionFlags = [`--${opt.name}`];
                if (opt.short) optionFlags.push(`-${opt.short}`);
                
                for (const flag of optionFlags) {
                  scriptParts.push(`                        "${flag}" {`);
                  
                  // Handle different value types
                  this.addPowerShellCompletionForType(scriptParts, opt.valueType, 24);
                  
                  scriptParts.push(`                            return`);
                  scriptParts.push(`                        }`);
                }
              }
            }
            
            scriptParts.push(`                    }`);
          }
          
          // Handle positional arguments
          if (subcmd.args && subcmd.args.length > 0) {
            scriptParts.push(`                    # Handle positional arguments`);
            scriptParts.push(`                    $argPosition = 0`);
            scriptParts.push(`                    $seenParams = @()`);
            scriptParts.push(`                    for ($i = 2; $i -lt $commandElements.Count; $i++) {`);
            scriptParts.push(`                        $element = $commandElements[$i].Extent.Text`);
            scriptParts.push(`                        if ($element -like "-*") {`);
            scriptParts.push(`                            $seenParams += $element`);
            scriptParts.push(`                            if (${this.generateOptionCheckCode(subcmd.options || [])}) {`);
            scriptParts.push(`                                $i++  # Skip the option's value`);
            scriptParts.push(`                            }`);
            scriptParts.push(`                        } else {`);
            scriptParts.push(`                            $argPosition++`);
            scriptParts.push(`                        }`);
            scriptParts.push(`                    }`);
            scriptParts.push(``);
            
            // Handle each argument by position
            scriptParts.push(`                    switch ($argPosition) {`);
            
            for (let i = 0; i < subcmd.args.length; i++) {
              const arg = subcmd.args[i];
              scriptParts.push(`                        ${i} {`);
              
              if (arg.completionType) {
                this.addPowerShellCompletionForType(scriptParts, arg.completionType, 28);
              }
              
              scriptParts.push(`                            return`);
              scriptParts.push(`                        }`);
            }
            
            scriptParts.push(`                    }`);
          }
          
          scriptParts.push(`                }`);
        }
        
        scriptParts.push(`            }`);
      } else {
        // Handle options for simple commands
        if (cmd.options && cmd.options.length > 0) {
          scriptParts.push(`            # Handle options for ${cmd.name}`);
          scriptParts.push(`            if ($wordToComplete -like "-*") {`);
          scriptParts.push(`                @(`);
          
          for (const opt of cmd.options) {
            const shortForm = opt.short ? `, '-${opt.short}'` : '';
            scriptParts.push(`                    [System.Management.Automation.CompletionResult]::new('--${opt.name}', '--${opt.name}'${shortForm}, 'ParameterValue', '${opt.description.replace(/'/g, "''")}')`);
          }
          
          scriptParts.push(`                ) | Where-Object { $_.CompletionText -like "$wordToComplete*" }`);
          scriptParts.push(`                return`);
          scriptParts.push(`            }`);
          scriptParts.push(``);
          
          // Handle option arguments
          scriptParts.push(`            # Handle option arguments`);
          scriptParts.push(`            $previousParameter = $commandElements[$commandElements.Count - 2].Extent.Text`);
          scriptParts.push(`            switch ($previousParameter) {`);
          
          for (const opt of cmd.options) {
            if (opt.takesValue && opt.valueType) {
              const optionFlags = [`--${opt.name}`];
              if (opt.short) optionFlags.push(`-${opt.short}`);
              
              for (const flag of optionFlags) {
                scriptParts.push(`                "${flag}" {`);
                
                // Handle different value types
                this.addPowerShellCompletionForType(scriptParts, opt.valueType, 20);
                
                scriptParts.push(`                    return`);
                scriptParts.push(`                }`);
              }
            }
          }
          
          scriptParts.push(`            }`);
        }
        
        // Handle positional arguments
        if (cmd.args && cmd.args.length > 0) {
          scriptParts.push(`            # Handle positional arguments`);
          scriptParts.push(`            $argPosition = 0`);
          scriptParts.push(`            $seenParams = @()`);
          scriptParts.push(`            for ($i = 1; $i -lt $commandElements.Count; $i++) {`);
          scriptParts.push(`                $element = $commandElements[$i].Extent.Text`);
          scriptParts.push(`                if ($element -like "-*") {`);
          scriptParts.push(`                    $seenParams += $element`);
          scriptParts.push(`                    if (${this.generateOptionCheckCode(cmd.options || [])}) {`);
          scriptParts.push(`                        $i++  # Skip the option's value`);
          scriptParts.push(`                    }`);
          scriptParts.push(`                } else {`);
          scriptParts.push(`                    $argPosition++`);
          scriptParts.push(`                }`);
          scriptParts.push(`            }`);
          scriptParts.push(``);
          
          // Handle each argument by position
          scriptParts.push(`            switch ($argPosition) {`);
          
          for (let i = 0; i < cmd.args.length; i++) {
            const arg = cmd.args[i];
            scriptParts.push(`                ${i} {`);
            
            if (arg.completionType) {
              this.addPowerShellCompletionForType(scriptParts, arg.completionType, 20);
            }
            
            scriptParts.push(`                    return`);
            scriptParts.push(`                }`);
          }
          
          scriptParts.push(`            }`);
        }
      }
      
      scriptParts.push(`        }`);
    }
    
    scriptParts.push(`    }`);
    scriptParts.push(`}`);
    
    return scriptParts.join('\n');
  }
  
  /**
   * Generate code to check if an option takes a value
   * 
   * @param options List of command options
   * @returns PowerShell code to check if an option takes a value
   */
  private generateOptionCheckCode(options: CompletionOption[]): string {
    const optionsWithValues = options
      .filter(opt => opt.takesValue)
      .map(opt => {
        const checks = [];
        if (opt.short) checks.push(`$element -eq "-${opt.short}"`);
        checks.push(`$element -eq "--${opt.name}"`);
        return `(${checks.join(' -or ')})`;
      });
    
    if (optionsWithValues.length === 0) {
      return '$false';
    }
    
    return optionsWithValues.join(' -or ');
  }
  
  /**
   * Add PowerShell-specific completion code for a value type
   * 
   * @param scriptParts Array of script lines
   * @param valueType Type of value to complete
   * @param indent Indentation level
   */
  private addPowerShellCompletionForType(scriptParts: string[], valueType: string | undefined, indent: number): void {
    if (!valueType) return;
    
    const indentStr = ' '.repeat(indent);
    
    switch (valueType) {
      case 'file':
        scriptParts.push(`${indentStr}Get-ChildItem -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {`);
        scriptParts.push(`${indentStr}    [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ParameterValue', $_.Name)`);
        scriptParts.push(`${indentStr}}`);
        break;
      case 'directory':
        scriptParts.push(`${indentStr}Get-ChildItem -Directory | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {`);
        scriptParts.push(`${indentStr}    [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ParameterValue', $_.Name)`);
        scriptParts.push(`${indentStr}}`);
        break;
      case 'provider':
        scriptParts.push(`${indentStr}Get-EmberDynamicCompletions -type provider -filter $wordToComplete | ForEach-Object {`);
        scriptParts.push(`${indentStr}    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)`);
        scriptParts.push(`${indentStr}}`);
        break;
      case 'model':
        scriptParts.push(`${indentStr}Get-EmberDynamicCompletions -type model -filter $wordToComplete | ForEach-Object {`);
        scriptParts.push(`${indentStr}    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)`);
        scriptParts.push(`${indentStr}}`);
        break;
      case 'template':
        scriptParts.push(`${indentStr}Get-EmberDynamicCompletions -type template -filter $wordToComplete | ForEach-Object {`);
        scriptParts.push(`${indentStr}    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)`);
        scriptParts.push(`${indentStr}}`);
        break;
      default:
        // Custom completion function would go here
        break;
    }
  }
}

/**
 * Completion Generator Factory
 * Creates the appropriate generator based on shell type
 */
class CompletionGeneratorFactory {
  /**
   * Get a shell-specific completion generator
   * 
   * @param shell Shell type
   * @returns Shell-specific generator
   */
  static getGenerator(shell: 'bash' | 'zsh' | 'fish' | 'powershell'): ShellCompletionGenerator {
    switch (shell) {
      case 'bash':
        return new BashCompletionGenerator();
      case 'zsh':
        return new ZshCompletionGenerator();
      case 'fish':
        return new FishCompletionGenerator();
      case 'powershell':
        return new PowerShellCompletionGenerator();
      default:
        throw new Error(`Unsupported shell: ${shell}`);
    }
  }
}

/**
 * Generate completion command specification
 * @returns Command specification for the completion command
 */
function getCompletionCommandSpec(): CompletionCommand {
  return {
    name: 'completion',
    description: 'Generate shell completion scripts',
    subcommands: [
      {
        name: 'bash',
        description: 'Generate Bash completion script',
        args: [
          {
            name: 'output',
            description: 'Output file (defaults to stdout)',
            required: false,
            completionType: 'file'
          }
        ]
      },
      {
        name: 'zsh',
        description: 'Generate Zsh completion script',
        args: [
          {
            name: 'output',
            description: 'Output file (defaults to stdout)',
            required: false,
            completionType: 'file'
          }
        ]
      },
      {
        name: 'fish',
        description: 'Generate Fish completion script',
        args: [
          {
            name: 'output',
            description: 'Output file (defaults to stdout)',
            required: false,
            completionType: 'file'
          }
        ]
      },
      {
        name: 'powershell',
        description: 'Generate PowerShell completion script',
        args: [
          {
            name: 'output',
            description: 'Output file (defaults to stdout)',
            required: false,
            completionType: 'file'
          }
        ]
      },
      {
        name: 'install',
        description: 'Install completion script for the current shell',
        options: [
          {
            name: 'force',
            short: 'f',
            description: 'Overwrite existing completion script',
          }
        ]
      }
    ]
  };
}

/**
 * Command specifications for the Ember CLI
 * This defines the structure of all available commands, which is used
 * to generate completion scripts for different shells.
 */
const emberCommandSpecs: CompletionCommand[] = [
  {
    name: 'version',
    description: 'Display version information',
    options: [
      {
        name: 'check',
        description: 'Check for updates',
      }
    ]
  },
  {
    name: 'provider',
    description: 'Manage LLM providers',
    subcommands: [
      {
        name: 'list',
        description: 'List all available LLM providers',
      },
      {
        name: 'configure',
        description: 'Configure a provider with API keys',
        args: [
          {
            name: 'provider',
            description: 'Provider ID',
            required: true,
            completionType: 'provider'
          }
        ],
        options: [
          {
            name: 'key',
            short: 'k',
            description: 'API key (or omit to be prompted)',
            takesValue: true
          },
          {
            name: 'force',
            short: 'f',
            description: 'Overwrite existing configuration',
          }
        ]
      },
      {
        name: 'info',
        description: 'Display information about a provider',
        args: [
          {
            name: 'provider',
            description: 'Provider ID',
            required: true,
            completionType: 'provider'
          }
        ]
      },
      {
        name: 'use',
        description: 'Set a provider as default',
        args: [
          {
            name: 'provider',
            description: 'Provider ID',
            required: true,
            completionType: 'provider'
          }
        ]
      }
    ]
  },
  {
    name: 'model',
    description: 'Manage LLM models',
    subcommands: [
      {
        name: 'list',
        description: 'List all available LLM models',
        options: [
          {
            name: 'provider',
            short: 'p',
            description: 'Filter models by provider',
            takesValue: true,
            valueType: 'provider'
          }
        ]
      },
      {
        name: 'info',
        description: 'Display information about a model',
        args: [
          {
            name: 'model',
            description: 'Model ID',
            required: true,
            completionType: 'model'
          }
        ]
      },
      {
        name: 'use',
        description: 'Set a model as default',
        args: [
          {
            name: 'model',
            description: 'Model ID',
            required: true,
            completionType: 'model'
          }
        ]
      },
      {
        name: 'benchmark',
        description: 'Run benchmark tests on a model',
        args: [
          {
            name: 'model',
            description: 'Model ID',
            required: true,
            completionType: 'model'
          }
        ],
        options: [
          {
            name: 'tests',
            short: 't',
            description: 'Number of tests to run',
            takesValue: true
          },
          {
            name: 'concurrency',
            short: 'c',
            description: 'Concurrency level',
            takesValue: true
          }
        ]
      }
    ]
  },
  {
    name: 'invoke',
    description: 'Invoke a model with a prompt',
    options: [
      {
        name: 'model',
        short: 'm',
        description: 'Model ID to use',
        takesValue: true,
        valueType: 'model'
      },
      {
        name: 'prompt',
        short: 'p',
        description: 'Prompt text to send to the model',
        takesValue: true
      },
      {
        name: 'file',
        short: 'f',
        description: 'Read prompt from file',
        takesValue: true,
        valueType: 'file'
      },
      {
        name: 'system',
        short: 's',
        description: 'System prompt (for chat models)',
        takesValue: true
      },
      {
        name: 'temperature',
        short: 't',
        description: 'Temperature setting (0.0-2.0)',
        takesValue: true
      },
      {
        name: 'show-usage',
        short: 'u',
        description: 'Show token usage statistics'
      },
      {
        name: 'output',
        short: 'o',
        description: 'Save output to file',
        takesValue: true,
        valueType: 'file'
      },
      {
        name: 'raw',
        short: 'r',
        description: 'Display raw output without formatting'
      },
      {
        name: 'stream',
        description: 'Stream the response token by token'
      }
    ]
  },
  {
    name: 'project',
    description: 'Manage Ember projects',
    subcommands: [
      {
        name: 'new',
        description: 'Create a new Ember project',
        args: [
          {
            name: 'name',
            description: 'Name of the project',
            required: true
          }
        ],
        options: [
          {
            name: 'template',
            short: 't',
            description: 'Project template',
            takesValue: true,
            valueType: 'template'
          },
          {
            name: 'directory',
            short: 'd',
            description: 'Project directory (defaults to name)',
            takesValue: true,
            valueType: 'directory'
          },
          {
            name: 'provider',
            short: 'p',
            description: 'Default provider to use',
            takesValue: true,
            valueType: 'provider'
          },
          {
            name: 'model',
            short: 'm',
            description: 'Default model to use',
            takesValue: true,
            valueType: 'model'
          }
        ]
      },
      {
        name: 'templates',
        description: 'List available project templates'
      },
      {
        name: 'analyze',
        description: 'Analyze an Ember project',
        args: [
          {
            name: 'directory',
            description: 'Project directory',
            required: false,
            completionType: 'directory'
          }
        ]
      }
    ]
  },
  {
    name: 'config',
    description: 'Manage Ember CLI configuration',
    subcommands: [
      {
        name: 'list',
        description: 'List all configuration settings',
        options: [
          {
            name: 'show-keys',
            description: 'Show API keys (not recommended)'
          }
        ]
      },
      {
        name: 'set',
        description: 'Set a configuration value',
        args: [
          {
            name: 'key',
            description: 'Configuration key',
            required: true
          },
          {
            name: 'value',
            description: 'Configuration value',
            required: true
          }
        ]
      },
      {
        name: 'get',
        description: 'Get a configuration value',
        args: [
          {
            name: 'key',
            description: 'Configuration key',
            required: true
          }
        ]
      },
      {
        name: 'reset',
        description: 'Reset all configuration to defaults',
        options: [
          {
            name: 'force',
            short: 'f',
            description: 'Skip confirmation'
          }
        ]
      },
      {
        name: 'export',
        description: 'Export configuration to a file',
        args: [
          {
            name: 'file',
            description: 'File path',
            required: true,
            completionType: 'file'
          }
        ]
      },
      {
        name: 'import',
        description: 'Import configuration from a file',
        args: [
          {
            name: 'file',
            description: 'File path',
            required: true,
            completionType: 'file'
          }
        ],
        options: [
          {
            name: 'force',
            short: 'f',
            description: 'Overwrite existing configuration'
          }
        ]
      },
      {
        name: 'usage-tracking',
        description: 'Enable or disable usage tracking',
        args: [
          {
            name: 'enabled',
            description: 'Whether usage tracking is enabled (true/false)',
            required: true
          }
        ]
      }
    ]
  },
  // Add completion command
  getCompletionCommandSpec()
];

/**
 * Generate a shell completion script
 * 
 * @param shell Shell type
 * @param commands Command specifications
 * @param programName Program name (default: ember)
 * @returns Completion script as a string
 */
export function generateCompletionScript(
  shell: 'bash' | 'zsh' | 'fish' | 'powershell',
  commands: CompletionCommand[] = emberCommandSpecs,
  programName: string = 'ember'
): string {
  const generator = CompletionGeneratorFactory.getGenerator(shell);
  return generator.generate(commands, programName);
}

/**
 * Write a completion script to a file or stdout
 * 
 * @param shell Shell type
 * @param outputPath Output file path (or undefined for stdout)
 * @returns Success message or error
 */
export async function writeCompletionScript(
  shell: 'bash' | 'zsh' | 'fish' | 'powershell',
  outputPath?: string
): Promise<string> {
  try {
    const script = generateCompletionScript(shell);
    
    if (outputPath) {
      fs.writeFileSync(outputPath, script);
      return `Completion script for ${shell} written to ${outputPath}`;
    } else {
      // Write to stdout
      process.stdout.write(script);
      return '';
    }
  } catch (error: any) {
    return `Error generating completion script: ${error.message}`;
  }
}

/**
 * Install a completion script for the current shell
 * 
 * @param force Whether to overwrite existing completion script
 * @returns Success message or error
 */
export async function installCompletionScript(force: boolean = false): Promise<string> {
  // Detect current shell
  const shell = process.env.SHELL;
  
  if (!shell) {
    return 'Could not detect shell. Please specify a shell type.';
  }
  
  let shellType: 'bash' | 'zsh' | 'fish' | 'powershell';
  let installPath: string;
  
  // Determine shell type and install path
  if (shell.includes('bash')) {
    shellType = 'bash';
    installPath = path.join(os.homedir(), '.bash_completion.d', 'ember');
  } else if (shell.includes('zsh')) {
    shellType = 'zsh';
    installPath = path.join(os.homedir(), '.zsh', 'completion', '_ember');
  } else if (shell.includes('fish')) {
    shellType = 'fish';
    installPath = path.join(os.homedir(), '.config', 'fish', 'completions', 'ember.fish');
  } else {
    return `Unsupported shell: ${shell}. Please use bash, zsh, or fish.`;
  }
  
  // Create directory if it doesn't exist
  const installDir = path.dirname(installPath);
  if (!fs.existsSync(installDir)) {
    fs.mkdirSync(installDir, { recursive: true });
  }
  
  // Check if file exists and handle force flag
  if (fs.existsSync(installPath) && !force) {
    return `Completion script already exists at ${installPath}. Use --force to overwrite.`;
  }
  
  // Generate and write completion script
  const script = generateCompletionScript(shellType);
  fs.writeFileSync(installPath, script);
  
  // Generate instructions for enabling completions
  let instructions = '';
  
  if (shellType === 'bash') {
    instructions = `\nTo enable completions, add the following line to your ~/.bashrc file:
source ~/.bash_completion.d/ember`;
  } else if (shellType === 'zsh') {
    instructions = `\nTo enable completions, add the following to your ~/.zshrc file:
fpath=(~/.zsh/completion $fpath)
autoload -U compinit
compinit`;
  } else if (shellType === 'fish') {
    instructions = `\nCompletions for fish are automatically loaded.`;
  }
  
  return `Completion script installed for ${shellType} at ${installPath}.${instructions}`;
}

/**
 * Command handler for the completion command
 * 
 * @param args Command arguments
 * @returns Result or message
 */
export async function handleCompletionCommand(args: any): Promise<string> {
  if (args.install) {
    return installCompletionScript(args.force);
  } else if (args.shell) {
    const shell = args.shell.toLowerCase();
    if (['bash', 'zsh', 'fish', 'powershell'].includes(shell)) {
      return writeCompletionScript(shell, args.output);
    } else {
      return `Unsupported shell: ${shell}. Supported shells: bash, zsh, fish, powershell.`;
    }
  } else {
    return 'Please specify a shell type or use the install option.';
  }
}

/**
 * Register the completion command with the CLI program
 * 
 * @param program The commander program instance
 */
export function registerCompletionCommand(program: any): void {
  const completionCommand = program
    .command('completion')
    .description('Generate shell completion scripts');
  
  // Shell-specific subcommands
  completionCommand
    .command('bash [output]')
    .description('Generate Bash completion script')
    .action((output) => {
      writeCompletionScript('bash', output)
        .then(msg => {
          if (msg) console.log(msg);
        })
        .catch(error => {
          console.error(chalk.red('Error:'), error.message);
        });
    });
  
  completionCommand
    .command('zsh [output]')
    .description('Generate Zsh completion script')
    .action((output) => {
      writeCompletionScript('zsh', output)
        .then(msg => {
          if (msg) console.log(msg);
        })
        .catch(error => {
          console.error(chalk.red('Error:'), error.message);
        });
    });
  
  completionCommand
    .command('fish [output]')
    .description('Generate Fish completion script')
    .action((output) => {
      writeCompletionScript('fish', output)
        .then(msg => {
          if (msg) console.log(msg);
        })
        .catch(error => {
          console.error(chalk.red('Error:'), error.message);
        });
    });
  
  completionCommand
    .command('powershell [output]')
    .description('Generate PowerShell completion script')
    .action((output) => {
      writeCompletionScript('powershell', output)
        .then(msg => {
          if (msg) console.log(msg);
        })
        .catch(error => {
          console.error(chalk.red('Error:'), error.message);
        });
    });
  
  // Installation command
  completionCommand
    .command('install')
    .description('Install completion script for the current shell')
    .option('-f, --force', 'Overwrite existing completion script')
    .action((options) => {
      installCompletionScript(options.force)
        .then(msg => {
          console.log(msg);
        })
        .catch(error => {
          console.error(chalk.red('Error:'), error.message);
        });
    });
}