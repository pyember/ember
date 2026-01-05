/**
 * Configuration management utilities for the setup wizard.
 *
 * Delegate persistence to the Python bridge (`ember.cli.commands.configure_api`)
 * so the wizard does not reimplement config/credential semantics.
 */

import {execa} from 'execa';

export interface Config {
  version: string;
  default_provider?: string;
  providers: {
    [provider: string]: {
      default_model?: string;
      organization_id?: string;
      base_url?: string;
    };
  };
}

export async function saveCredentials(provider: string, apiKey: string): Promise<void> {
  await execa(
    'python',
    ['-m', 'ember.cli.commands.configure_api', 'save-key', provider],
    {input: apiKey, reject: true},
  );
}

export async function saveConfig(updates: Partial<Config>): Promise<void> {
  await execa('python', ['-m', 'ember.cli.commands.configure_api', 'save-config'], {
    input: JSON.stringify(updates),
    reject: true,
  });
}
