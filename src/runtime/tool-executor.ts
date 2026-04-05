/**
 * ToolExecutor — centralized tool layer for all runtimes.
 *
 * This is the canonical tool layer. All runtimes should route through it
 * long-term. Claude's built-in tool bypass is temporary.
 *
 * Tool categories:
 *   container_mandatory: bash, write, edit — always via ContainerManager
 *   container_preferred: read, glob, grep — container if available, host fallback
 *   host_only: send_message, schedule_task, etc. — direct host execution (IPC)
 *   either: web_search, web_fetch — host or container
 */
import fs from 'fs';
import path from 'path';

import { DATA_DIR } from '../config.js';
import { resolveGroupIpcPath } from '../group-folder.js';
import { logger } from '../logger.js';

import type {
  ContainerManager,
  RuntimeId,
  ToolCall,
  ToolDefinition,
  ToolExecutor as IToolExecutor,
  ToolResult,
} from './types.js';

// --- Tool category routing ---

type ToolCategory =
  | 'container_mandatory'
  | 'container_preferred'
  | 'host_only'
  | 'either';

const TOOL_CATEGORIES: Record<string, ToolCategory> = {
  bash: 'container_mandatory',
  write: 'container_mandatory',
  edit: 'container_mandatory',
  read: 'container_preferred',
  glob: 'container_preferred',
  grep: 'container_preferred',
  send_message: 'host_only',
  schedule_task: 'host_only',
  list_tasks: 'host_only',
  pause_task: 'host_only',
  resume_task: 'host_only',
  cancel_task: 'host_only',
  update_task: 'host_only',
  register_group: 'host_only',
  list_skills: 'host_only',
  load_skill: 'host_only',
  web_search: 'either',
  web_fetch: 'either',
};

// --- IPC file helper (same as ipc-mcp-stdio.ts) ---

function writeIpcFile(dir: string, data: object): string {
  fs.mkdirSync(dir, { recursive: true });
  const filename = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`;
  const filepath = path.join(dir, filename);
  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
  fs.renameSync(tempPath, filepath);
  return filename;
}

// --- Tool definitions ---

const HOST_TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    name: 'send_message',
    description:
      'Send a message to the user or group immediately. Use for progress updates or multiple messages.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'The message text to send' },
        sender: {
          type: 'string',
          description: 'Your role/identity name (optional)',
        },
      },
      required: ['text'],
    },
    execution: 'host',
  },
  {
    name: 'schedule_task',
    description: 'Schedule a recurring or one-time task.',
    parameters: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'What the agent should do' },
        schedule_type: {
          type: 'string',
          enum: ['cron', 'interval', 'once'],
        },
        schedule_value: { type: 'string', description: 'Schedule expression' },
        context_mode: {
          type: 'string',
          enum: ['group', 'isolated'],
          default: 'group',
        },
        target_group_jid: {
          type: 'string',
          description: 'Target group JID (main only)',
        },
        script: {
          type: 'string',
          description: 'Optional bash script to run before waking agent',
        },
      },
      required: ['prompt', 'schedule_type', 'schedule_value'],
    },
    execution: 'host',
  },
  {
    name: 'list_tasks',
    description: 'List all scheduled tasks.',
    parameters: { type: 'object', properties: {} },
    execution: 'host',
  },
  {
    name: 'pause_task',
    description: 'Pause a scheduled task.',
    parameters: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID to pause' },
      },
      required: ['task_id'],
    },
    execution: 'host',
  },
  {
    name: 'resume_task',
    description: 'Resume a paused task.',
    parameters: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID to resume' },
      },
      required: ['task_id'],
    },
    execution: 'host',
  },
  {
    name: 'cancel_task',
    description: 'Cancel and delete a scheduled task.',
    parameters: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID to cancel' },
      },
      required: ['task_id'],
    },
    execution: 'host',
  },
  {
    name: 'update_task',
    description: 'Update an existing scheduled task.',
    parameters: {
      type: 'object',
      properties: {
        task_id: { type: 'string', description: 'Task ID to update' },
        prompt: { type: 'string' },
        schedule_type: { type: 'string', enum: ['cron', 'interval', 'once'] },
        schedule_value: { type: 'string' },
        script: { type: 'string' },
      },
      required: ['task_id'],
    },
    execution: 'host',
  },
  {
    name: 'register_group',
    description: 'Register a new chat/group (main only).',
    parameters: {
      type: 'object',
      properties: {
        jid: { type: 'string', description: 'Chat JID' },
        name: { type: 'string', description: 'Display name' },
        folder: { type: 'string', description: 'Channel-prefixed folder name' },
        trigger: { type: 'string', description: 'Trigger word' },
        requiresTrigger: { type: 'boolean' },
      },
      required: ['jid', 'name', 'folder', 'trigger'],
    },
    execution: 'host',
  },
  {
    name: 'list_skills',
    description: 'List all available skills with their names and descriptions.',
    parameters: { type: 'object', properties: {} },
    execution: 'host',
  },
  {
    name: 'load_skill',
    description:
      'Load a skill by name. Returns the full instructions and a list of included files (templates, examples). Use this when you need detailed instructions for a specific capability.',
    parameters: {
      type: 'object',
      properties: {
        skill_name: {
          type: 'string',
          description: 'Name of the skill to load',
        },
      },
      required: ['skill_name'],
    },
    execution: 'host',
  },
];

const CONTAINER_TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    name: 'bash',
    description: 'Execute a bash command in the workspace',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The bash command to execute' },
        timeout: {
          type: 'number',
          description: 'Timeout in ms (default 120000)',
        },
      },
      required: ['command'],
    },
    execution: 'container',
  },
  {
    name: 'read',
    description: 'Read a file from the workspace',
    parameters: {
      type: 'object',
      properties: {
        file_path: { type: 'string', description: 'Path to the file' },
        offset: { type: 'number', description: 'Line number to start from' },
        limit: { type: 'number', description: 'Number of lines to read' },
      },
      required: ['file_path'],
    },
    execution: 'container',
  },
  {
    name: 'write',
    description: 'Write a file to the workspace',
    parameters: {
      type: 'object',
      properties: {
        file_path: { type: 'string', description: 'Path to write to' },
        content: { type: 'string', description: 'File content' },
      },
      required: ['file_path', 'content'],
    },
    execution: 'container',
  },
  {
    name: 'edit',
    description: 'Edit a file by replacing a string',
    parameters: {
      type: 'object',
      properties: {
        file_path: { type: 'string', description: 'Path to the file' },
        old_string: { type: 'string', description: 'Text to find' },
        new_string: { type: 'string', description: 'Replacement text' },
      },
      required: ['file_path', 'old_string', 'new_string'],
    },
    execution: 'container',
  },
  {
    name: 'glob',
    description: 'Find files by pattern',
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'Glob pattern' },
        path: { type: 'string', description: 'Directory to search in' },
      },
      required: ['pattern'],
    },
    execution: 'container',
  },
  {
    name: 'grep',
    description: 'Search file contents with regex',
    parameters: {
      type: 'object',
      properties: {
        pattern: { type: 'string', description: 'Regex pattern' },
        path: { type: 'string', description: 'File or directory to search' },
        include: { type: 'string', description: 'File glob filter' },
      },
      required: ['pattern'],
    },
    execution: 'container',
  },
  {
    name: 'web_search',
    description: 'Search the web',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
      },
      required: ['query'],
    },
    execution: 'either',
  },
  {
    name: 'web_fetch',
    description: 'Fetch a web page',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'URL to fetch' },
      },
      required: ['url'],
    },
    execution: 'either',
  },
];

// --- DefaultToolExecutor ---

export class DefaultToolExecutor implements IToolExecutor {
  constructor(private containerManager: ContainerManager) {}

  getTools(opts: { runtime: RuntimeId; isMain: boolean }): ToolDefinition[] {
    const tools = [...HOST_TOOL_DEFINITIONS, ...CONTAINER_TOOL_DEFINITIONS];
    // Non-main groups can't register groups
    if (!opts.isMain) {
      return tools.filter((t) => t.name !== 'register_group');
    }
    return tools;
  }

  async execute(call: ToolCall): Promise<ToolResult> {
    const category = TOOL_CATEGORIES[call.name];

    if (!category) {
      return { content: `Unknown tool: ${call.name}`, isError: true };
    }

    switch (category) {
      case 'host_only':
        return this.executeHostTool(call);

      case 'container_mandatory':
      case 'container_preferred':
        // Delegate to ContainerManager for sandboxed execution
        return this.containerManager.executeInContainer(call);

      case 'either':
        // Prefer host execution (simpler, no container overhead)
        if (call.context.containerId) {
          return this.containerManager.executeInContainer(call);
        }
        return this.executeHostTool(call);

      default:
        return {
          content: `Unhandled tool category for: ${call.name}`,
          isError: true,
        };
    }
  }

  // --- Host-side IPC tool implementations ---
  // These mirror the logic in container/agent-runner/src/ipc-mcp-stdio.ts
  // but execute directly on the host. The IPC watcher processes the
  // resulting files identically regardless of who wrote them.

  private executeHostTool(call: ToolCall): ToolResult {
    const { name, arguments: args, context } = call;
    const ipcDir = resolveGroupIpcPath(context.groupFolder);
    const messagesDir = path.join(ipcDir, 'messages');
    const tasksDir = path.join(ipcDir, 'tasks');

    switch (name) {
      case 'send_message': {
        const text = args.text as string;
        writeIpcFile(messagesDir, {
          type: 'message',
          chatJid: context.chatJid,
          text,
          sender: (args.sender as string) || undefined,
          groupFolder: context.groupFolder,
          timestamp: new Date().toISOString(),
        });
        return { content: 'Message sent.' };
      }

      case 'schedule_task': {
        const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        writeIpcFile(tasksDir, {
          type: 'schedule_task',
          taskId,
          prompt: args.prompt as string,
          script: (args.script as string) || undefined,
          schedule_type: args.schedule_type as string,
          schedule_value: args.schedule_value as string,
          context_mode: (args.context_mode as string) || 'group',
          targetJid:
            context.isMain && args.target_group_jid
              ? (args.target_group_jid as string)
              : context.chatJid,
          createdBy: context.groupFolder,
          timestamp: new Date().toISOString(),
        });
        return {
          content: `Task ${taskId} scheduled: ${args.schedule_type} - ${args.schedule_value}`,
        };
      }

      case 'list_tasks': {
        const tasksFile = path.join(ipcDir, 'current_tasks.json');
        try {
          if (!fs.existsSync(tasksFile)) {
            return { content: 'No scheduled tasks found.' };
          }
          const allTasks = JSON.parse(fs.readFileSync(tasksFile, 'utf-8'));
          const filtered = context.isMain
            ? allTasks
            : allTasks.filter(
                (t: { groupFolder: string }) =>
                  t.groupFolder === context.groupFolder,
              );
          if (filtered.length === 0) {
            return { content: 'No scheduled tasks found.' };
          }
          const formatted = filtered
            .map(
              (t: {
                id: string;
                prompt: string;
                schedule_type: string;
                schedule_value: string;
                status: string;
                next_run: string;
              }) =>
                `- [${t.id}] ${t.prompt.slice(0, 50)}... (${t.schedule_type}: ${t.schedule_value}) - ${t.status}, next: ${t.next_run || 'N/A'}`,
            )
            .join('\n');
          return { content: `Scheduled tasks:\n${formatted}` };
        } catch (err) {
          return {
            content: `Error reading tasks: ${err instanceof Error ? err.message : String(err)}`,
            isError: true,
          };
        }
      }

      case 'pause_task':
      case 'resume_task':
      case 'cancel_task': {
        writeIpcFile(tasksDir, {
          type: name,
          taskId: args.task_id as string,
          groupFolder: context.groupFolder,
          isMain: context.isMain,
          timestamp: new Date().toISOString(),
        });
        return {
          content: `Task ${args.task_id} ${name.replace('_task', '')} requested.`,
        };
      }

      case 'update_task': {
        const data: Record<string, unknown> = {
          type: 'update_task',
          taskId: args.task_id as string,
          groupFolder: context.groupFolder,
          isMain: String(context.isMain),
          timestamp: new Date().toISOString(),
        };
        if (args.prompt !== undefined) data.prompt = args.prompt;
        if (args.script !== undefined) data.script = args.script;
        if (args.schedule_type !== undefined)
          data.schedule_type = args.schedule_type;
        if (args.schedule_value !== undefined)
          data.schedule_value = args.schedule_value;
        writeIpcFile(tasksDir, data);
        return { content: `Task ${args.task_id} update requested.` };
      }

      case 'register_group': {
        if (!context.isMain) {
          return {
            content: 'Only the main group can register new groups.',
            isError: true,
          };
        }
        writeIpcFile(tasksDir, {
          type: 'register_group',
          jid: args.jid as string,
          name: args.name as string,
          folder: args.folder as string,
          trigger: args.trigger as string,
          requiresTrigger: (args.requiresTrigger as boolean) ?? false,
          timestamp: new Date().toISOString(),
        });
        return {
          content: `Group "${args.name}" registered. It will start receiving messages immediately.`,
        };
      }

      case 'list_skills': {
        const skills = this.listSkills();
        if (skills.length === 0) {
          return { content: 'No skills installed.' };
        }
        const formatted = skills
          .map((s) => `- ${s.name}: ${s.description}`)
          .join('\n');
        return { content: `Available skills:\n${formatted}` };
      }

      case 'load_skill': {
        const skillName = args.skill_name as string;
        const result = this.loadSkill(skillName);
        if (!result) {
          return {
            content: `Skill not found: ${skillName}`,
            isError: true,
          };
        }
        let content = result.instructions;
        if (result.files.length > 0) {
          content +=
            '\n\n--- Included files (use read tool to access) ---\n' +
            result.files.map((f) => `- ${f}`).join('\n');
        }
        return { content };
      }

      case 'web_search':
        // TODO: Implement host-side web search (fetch + parse)
        return {
          content: 'web_search not yet implemented on host',
          isError: true,
        };

      case 'web_fetch':
        // TODO: Implement host-side web fetch (fetch + HTML-to-text)
        return {
          content: 'web_fetch not yet implemented on host',
          isError: true,
        };

      default:
        return { content: `Host tool not implemented: ${name}`, isError: true };
    }
  }

  // --- Skill discovery ---

  private getSkillsDir(): string {
    return path.join(process.cwd(), 'container', 'skills');
  }

  private listSkills(): Array<{ name: string; description: string }> {
    const skillsDir = this.getSkillsDir();
    if (!fs.existsSync(skillsDir)) return [];

    const skills: Array<{ name: string; description: string }> = [];
    for (const entry of fs.readdirSync(skillsDir)) {
      const skillMd = path.join(skillsDir, entry, 'SKILL.md');
      if (!fs.existsSync(skillMd)) continue;
      const content = fs.readFileSync(skillMd, 'utf-8');
      const match = content.match(/^---\s*\n([\s\S]*?)\n---/);
      if (!match) continue;
      const frontmatter = match[1];
      const name = frontmatter.match(/^name:\s*(.+)$/m)?.[1]?.trim() || entry;
      const description =
        frontmatter.match(/^description:\s*(.+)$/m)?.[1]?.trim() || '';
      if (name) {
        skills.push({ name, description });
      }
    }
    return skills;
  }

  private loadSkill(
    skillName: string,
  ): { instructions: string; files: string[] } | null {
    const skillsDir = this.getSkillsDir();

    // Find the skill directory (match by folder name or frontmatter name)
    for (const entry of fs.readdirSync(skillsDir)) {
      const skillDir = path.join(skillsDir, entry);
      const skillMd = path.join(skillDir, 'SKILL.md');
      if (!fs.existsSync(skillMd)) continue;

      const content = fs.readFileSync(skillMd, 'utf-8');
      const match = content.match(/^---\s*\n([\s\S]*?)\n---/);
      const frontmatterName = match
        ? match[1].match(/^name:\s*(.+)$/m)?.[1]?.trim()
        : null;

      if (entry === skillName || frontmatterName === skillName) {
        // Collect non-SKILL.md files in the skill directory
        const files: string[] = [];
        const walk = (dir: string, prefix = '') => {
          for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
            const rel = prefix ? `${prefix}/${f.name}` : f.name;
            if (f.isDirectory()) {
              walk(path.join(dir, f.name), rel);
            } else if (f.name !== 'SKILL.md') {
              files.push(rel);
            }
          }
        };
        walk(skillDir);

        return { instructions: content, files };
      }
    }

    return null;
  }
}
