/**
 * Configuration Manager Service
 * 
 * Manages configuration for Ember CLI, including provider settings,
 * API keys, and user preferences.
 */

import Conf from 'conf';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';

/**
 * Config schema type
 */
interface ConfigSchema {
  providers: {
    [key: string]: {
      apiKey: string;
      [key: string]: any;
    };
  };
  defaultProvider: string | null;
  usage: {
    trackUsage: boolean;
    lastCheck: number;
  };
  settings: {
    [key: string]: any;
  };
}

/**
 * Configuration Manager
 * Uses the Singleton pattern to ensure a single instance
 */
export class ConfigManager {
  private static instance: ConfigManager;
  private config: Conf<ConfigSchema>;
  
  /**
   * Private constructor for Singleton pattern
   */
  private constructor() {
    // Create config store
    this.config = new Conf<ConfigSchema>({
      projectName: 'ember-cli',
      schema: {
        providers: {
          type: 'object',
          default: {}
        },
        defaultProvider: {
          type: ['string', 'null'],
          default: null
        },
        usage: {
          type: 'object',
          properties: {
            trackUsage: {
              type: 'boolean',
              default: true
            },
            lastCheck: {
              type: 'number',
              default: 0
            }
          },
          default: {
            trackUsage: true,
            lastCheck: 0
          }
        },
        settings: {
          type: 'object',
          default: {}
        }
      },
      encryptionKey: this.getEncryptionKey()
    });
    
    // Initialize config if needed
    this.initializeConfig();
  }
  
  /**
   * Get the singleton instance
   */
  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    
    return ConfigManager.instance;
  }
  
  /**
   * Get a provider configuration
   * 
   * @param providerId The provider ID
   * @returns The provider configuration or null if not found
   */
  public getProviderConfig(providerId: string): any {
    const providers = this.config.get('providers');
    return providers[providerId] || null;
  }
  
  /**
   * Check if a provider has configuration
   * 
   * @param providerId The provider ID
   * @returns True if the provider is configured
   */
  public hasProviderConfig(providerId: string): boolean {
    const providers = this.config.get('providers');
    return !!providers[providerId];
  }
  
  /**
   * Configure a provider
   * 
   * @param providerId The provider ID
   * @param config The provider configuration
   */
  public configureProvider(providerId: string, config: any): void {
    const providers = this.config.get('providers');
    
    providers[providerId] = {
      ...providers[providerId],
      ...config
    };
    
    this.config.set('providers', providers);
    
    // Also set environment variable for current session
    process.env[`${providerId.toUpperCase()}_API_KEY`] = config.apiKey;
  }
  
  /**
   * Remove a provider configuration
   * 
   * @param providerId The provider ID
   */
  public removeProviderConfig(providerId: string): void {
    const providers = this.config.get('providers');
    
    if (providers[providerId]) {
      delete providers[providerId];
      this.config.set('providers', providers);
    }
    
    // Check if it was the default provider
    const defaultProvider = this.config.get('defaultProvider');
    if (defaultProvider === providerId) {
      this.config.set('defaultProvider', null);
    }
  }
  
  /**
   * Get the default provider
   * 
   * @returns The default provider ID or null if not set
   */
  public getDefaultProvider(): string | null {
    return this.config.get('defaultProvider');
  }
  
  /**
   * Set the default provider
   * 
   * @param providerId The provider ID
   */
  public setDefaultProvider(providerId: string): void {
    this.config.set('defaultProvider', providerId);
  }
  
  /**
   * Get a setting value
   * 
   * @param key The setting key
   * @param defaultValue The default value if not found
   * @returns The setting value or default value
   */
  public getSetting<T>(key: string, defaultValue: T): T {
    const settings = this.config.get('settings');
    return (settings[key] as T) ?? defaultValue;
  }
  
  /**
   * Set a setting value
   * 
   * @param key The setting key
   * @param value The setting value
   */
  public setSetting<T>(key: string, value: T): void {
    const settings = this.config.get('settings');
    settings[key] = value;
    this.config.set('settings', settings);
  }
  
  /**
   * Get all settings
   * 
   * @returns All settings
   */
  public getAllSettings(): any {
    return this.config.get('settings');
  }
  
  /**
   * Check if usage tracking is enabled
   * 
   * @returns True if usage tracking is enabled
   */
  public isUsageTrackingEnabled(): boolean {
    const usage = this.config.get('usage');
    return usage.trackUsage;
  }
  
  /**
   * Set usage tracking
   * 
   * @param enabled True to enable usage tracking
   */
  public setUsageTracking(enabled: boolean): void {
    const usage = this.config.get('usage');
    usage.trackUsage = enabled;
    this.config.set('usage', usage);
  }
  
  /**
   * Record usage check
   */
  public recordUsageCheck(): void {
    const usage = this.config.get('usage');
    usage.lastCheck = Date.now();
    this.config.set('usage', usage);
  }
  
  /**
   * Reset all configuration
   */
  public resetConfig(): void {
    this.config.clear();
    this.initializeConfig();
  }
  
  /**
   * Export configuration to a file
   * 
   * @param filePath The file path to export to
   */
  public exportConfig(filePath: string): void {
    const data = this.config.store;
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  }
  
  /**
   * Import configuration from a file
   * 
   * @param filePath The file path to import from
   */
  public importConfig(filePath: string): void {
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    this.config.store = data;
  }
  
  /**
   * Initialize configuration with defaults
   */
  private initializeConfig(): void {
    // Ensure the providers object exists
    if (!this.config.has('providers')) {
      this.config.set('providers', {});
    }
    
    // Ensure the usage object exists
    if (!this.config.has('usage')) {
      this.config.set('usage', {
        trackUsage: true,
        lastCheck: 0
      });
    }
    
    // Ensure the settings object exists
    if (!this.config.has('settings')) {
      this.config.set('settings', {});
    }
  }
  
  /**
   * Get encryption key for sensitive data
   * Creates a stable key based on machine-specific information
   */
  private getEncryptionKey(): string {
    // Use a combination of machine-specific values for encryption
    const hostname = os.hostname();
    const username = os.userInfo().username;
    const cpus = os.cpus().length;
    
    // Create a stable key
    const input = `ember-cli:${hostname}:${username}:${cpus}`;
    return crypto.createHash('sha256').update(input).digest('hex').slice(0, 32);
  }
}