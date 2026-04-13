/**
 * Specialist persona utilities.
 *
 * Loads specialist personas from files or AGENT.md inline definitions.
 * Used by native subagent systems in each runtime:
 * - Claude: TeamCreate/SendMessage
 * - Codex: spawnAgent/sendInput/wait/closeAgent
 * - Gemini: ADK sub-agents
 */
import fs from 'fs';
import path from 'path';

const SPECIALISTS_DIR = '/workspace/group/specialists';
const AGENT_MD_PATH = '/workspace/group/AGENT.md';

/**
 * Load specialist persona by name.
 * First checks specialists/<name>.md, then falls back to parsing
 * the ## Specialists section of AGENT.md for an inline definition.
 */
export function loadSpecialistPersona(name: string): string | null {
  const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '');
  if (!safeName) return null;

  const personaFile = path.join(SPECIALISTS_DIR, `${safeName}.md`);
  if (fs.existsSync(personaFile)) {
    return fs.readFileSync(personaFile, 'utf-8');
  }

  if (fs.existsSync(AGENT_MD_PATH)) {
    const agentMd = fs.readFileSync(AGENT_MD_PATH, 'utf-8');
    const persona = extractInlineSpecialist(agentMd, safeName);
    if (persona) return persona;
  }

  return null;
}

function extractInlineSpecialist(markdown: string, name: string): string | null {
  const pattern = new RegExp(
    `###\\s+${name}\\s*\\n([\\s\\S]*?)(?=\\n###\\s|\\n##\\s|$)`,
    'i',
  );
  const match = markdown.match(pattern);
  return match ? match[1].trim() : null;
}

export function listSpecialists(): string[] {
  const names = new Set<string>();

  if (fs.existsSync(SPECIALISTS_DIR)) {
    for (const file of fs.readdirSync(SPECIALISTS_DIR)) {
      if (file.endsWith('.md')) {
        names.add(file.replace(/\.md$/, ''));
      }
    }
  }

  if (fs.existsSync(AGENT_MD_PATH)) {
    const content = fs.readFileSync(AGENT_MD_PATH, 'utf-8');
    const specialistsSection = content.match(
      /## Specialists\s*\n([\s\S]*?)(?=\n## |$)/i,
    );
    if (specialistsSection) {
      const headings = specialistsSection[1].matchAll(/###\s+(\w+)/g);
      for (const m of headings) {
        names.add(m[1].toLowerCase());
      }
    }
  }

  return [...names].sort();
}
