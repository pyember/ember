import { Listr, ListrTask } from 'listr2';
import { Spinner } from './components';
import { getOutputOptions } from './output-manager';

/**
 * Task representation
 */
export interface Task<Ctx extends Record<string, any> = Record<string, any>> {
  /** Task title */
  title: string;
  /** Task execution function */
  task: (ctx: Ctx, task: ListrTask) => Promise<any> | any;
  /** Whether to skip the task */
  skip?: (ctx: Ctx) => boolean | string | Promise<boolean | string>;
  /** Whether the task is enabled */
  enabled?: (ctx: Ctx) => boolean | Promise<boolean>;
  /** Subtasks */
  subtasks?: Task<Ctx>[];
  /** Whether to exit on error */
  exitOnError?: boolean;
}

/**
 * Task manager options
 */
export interface TaskManagerOptions<Ctx extends Record<string, any> = Record<string, any>> {
  /** Initial context value */
  initialContext?: Ctx;
  /** Whether to exit on error */
  exitOnError?: boolean;
  /** Whether to show subtasks */
  showSubtasks?: boolean;
  /** Task concurrency limit */
  concurrent?: boolean | number;
  /** Whether to collapse subtasks on completion */
  collapseSubtasks?: boolean;
  /** Whether to render output in JSON format */
  json?: boolean;
}

/**
 * Task manager class
 * Handles task execution with progress indicators
 */
export class TaskManager<Ctx extends Record<string, any> = Record<string, any>> {
  private tasks: Task<Ctx>[] = [];
  private options: TaskManagerOptions<Ctx>;
  private spinner: Spinner | null = null;
  
  /**
   * Creates a new task manager
   * @param options Task manager options
   */
  constructor(options: TaskManagerOptions<Ctx> = {}) {
    const outputOptions = getOutputOptions();
    
    this.options = {
      initialContext: {} as Ctx,
      exitOnError: true,
      showSubtasks: true,
      concurrent: false,
      collapseSubtasks: true,
      json: outputOptions.format === 'json',
      ...options,
    };
  }
  
  /**
   * Adds a task to the task manager
   * @param task Task to add
   * @returns this for chaining
   */
  add(task: Task<Ctx>): TaskManager<Ctx> {
    this.tasks.push(task);
    return this;
  }
  
  /**
   * Adds multiple tasks to the task manager
   * @param tasks Tasks to add
   * @returns this for chaining
   */
  addTasks(tasks: Task<Ctx>[]): TaskManager<Ctx> {
    this.tasks.push(...tasks);
    return this;
  }
  
  /**
   * Runs the tasks
   * @returns Context after task execution
   */
  async run(): Promise<Ctx> {
    // If running in quiet or JSON mode, use a simple spinner instead of Listr
    if (this.options.json) {
      return this.runSimple();
    }
    
    const runner = new Listr<Ctx>(this.mapTasks(this.tasks), {
      concurrent: this.options.concurrent,
      exitOnError: this.options.exitOnError,
      rendererOptions: {
        collapseSubtasks: this.options.collapseSubtasks,
        showSubtasks: this.options.showSubtasks,
        collapseErrors: false,
        formatOutput: 'wrap',
        removeEmptyLines: true,
      },
      ctx: this.options.initialContext,
    });
    
    try {
      return await runner.run();
    } catch (error) {
      if (this.options.exitOnError) {
        throw error;
      }
      return this.options.initialContext;
    }
  }
  
  /**
   * Runs tasks in simple mode (for JSON or quiet mode)
   * @returns Context after task execution
   */
  private async runSimple(): Promise<Ctx> {
    const context = { ...this.options.initialContext };
    
    for (const task of this.tasks) {
      // Skip task if skip function returns true or a message
      if (task.skip) {
        const skipResult = await task.skip(context);
        if (skipResult) {
          continue;
        }
      }
      
      // Skip task if enabled function returns false
      if (task.enabled !== undefined) {
        const enabled = await task.enabled(context);
        if (!enabled) {
          continue;
        }
      }
      
      // Start spinner (except in JSON mode)
      if (!this.options.json) {
        this.spinner = new Spinner(task.title).start();
      }
      
      try {
        // Create a mock task object for compatibility with Listr
        const mockTask = {
          title: task.title,
          output: '',
          // Mock additional methods
          newListr: () => mockTask,
          report: () => {},
          skip: () => {},
        } as unknown as ListrTask;
        
        // Execute the task
        await task.task(context, mockTask);
        
        // Stop spinner with success
        if (this.spinner) {
          this.spinner.succeed();
        }
        
        // Run subtasks if present
        if (task.subtasks && task.subtasks.length > 0) {
          for (const subtask of task.subtasks) {
            // Skip subtask if skip function returns true or a message
            if (subtask.skip) {
              const skipResult = await subtask.skip(context);
              if (skipResult) {
                continue;
              }
            }
            
            // Skip subtask if enabled function returns false
            if (subtask.enabled !== undefined) {
              const enabled = await subtask.enabled(context);
              if (!enabled) {
                continue;
              }
            }
            
            // Start spinner for subtask
            if (!this.options.json) {
              this.spinner = new Spinner(`  ${subtask.title}`).start();
            }
            
            try {
              await subtask.task(context, mockTask);
              
              // Stop spinner with success
              if (this.spinner) {
                this.spinner.succeed();
              }
            } catch (subtaskError) {
              // Stop spinner with error
              if (this.spinner) {
                this.spinner.fail();
              }
              
              if (subtask.exitOnError !== false && this.options.exitOnError) {
                throw subtaskError;
              }
            }
          }
        }
      } catch (error) {
        // Stop spinner with error
        if (this.spinner) {
          this.spinner.fail();
        }
        
        if (task.exitOnError !== false && this.options.exitOnError) {
          throw error;
        }
      }
    }
    
    return context;
  }
  
  /**
   * Maps ember Task interface to Listr Task interface
   * @param tasks Tasks to map
   * @returns Mapped Listr tasks
   */
  private mapTasks(tasks: Task<Ctx>[]): ListrTask<Ctx, any, any>[] {
    return tasks.map(task => ({
      title: task.title,
      task: task.task,
      skip: task.skip,
      enabled: task.enabled,
      exitOnError: task.exitOnError,
      options: { persistentOutput: true },
      subtasks: task.subtasks ? this.mapTasks(task.subtasks) : undefined,
    }));
  }
}

/**
 * Creates a new task manager
 * @param options Task manager options
 * @returns New task manager
 */
export function createTaskManager<Ctx extends Record<string, any> = Record<string, any>>(
  options?: TaskManagerOptions<Ctx>
): TaskManager<Ctx> {
  return new TaskManager<Ctx>(options);
}