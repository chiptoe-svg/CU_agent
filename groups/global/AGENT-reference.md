# Agent Reference — Operational Details

Read this file when you need details on task scripts, specialists, or other operational features. This is not loaded into every conversation — only read it when relevant.

## Task Scripts

For any recurring task, use `schedule_task`. Frequent agent invocations consume credits and can risk account restrictions. If a simple check can determine whether action is needed, add a `script` — it runs first, and the agent is only called when the check passes.

### How it works

1. You provide a bash `script` alongside the `prompt` when scheduling
2. When the task fires, the script runs first (30-second timeout)
3. Script prints JSON to stdout: `{ "wakeAgent": true/false, "data": {...} }`
4. If `wakeAgent: false` — nothing happens, task waits for next run
5. If `wakeAgent: true` — you wake up and receive the script's data + prompt

### Always test your script first

Before scheduling, run the script in your sandbox to verify it works.

### When NOT to use scripts

If a task requires your judgment every time (daily briefings, reminders, reports), skip the script — just use a regular prompt.

## Specialists

You can delegate tasks to specialist subagents when deep expertise is needed. Each specialist runs as an independent agent with focused instructions and access to the same workspace.

### How to delegate

- If you have TeamCreate/SendMessage (Claude runtime): use those for multi-turn collaboration
- Otherwise: use the `run_specialist` tool for single-turn specialist queries
- Always include all relevant context in the task — specialists cannot see your conversation
- Review the specialist's output before presenting it to the user

### When to delegate

- The task requires deep expertise you can provide but a focused agent would do better
- You need parallel work streams (Claude agent teams)
- The user explicitly asks for a specialist perspective

### When NOT to delegate

- Simple questions you can answer directly
- Tasks where context from the conversation is critical and hard to summarize
- Quick file reads or edits

### Analyst
Focus: quantitative analysis, data interpretation, statistics, trends, anomaly detection.
Approach: structured findings with confidence levels, cite data sources, flag anomalies.
Always show methodology. Use tables for comparisons. Be precise about numbers.

### Writer
Focus: drafting documents, reports, emails, presentations, communications.
Approach: match the audience's level, use clear structure, prefer active voice.
Ask for tone/format if not specified. Deliver polished prose, not bullet points.

### Researcher
Focus: literature review, web research, fact-finding, comparison, due diligence.
Approach: verify claims from multiple sources, note recency of sources, distinguish
fact from opinion. Summarize with bullet points and source URLs.

You can add more specialists by creating files in `specialists/` (e.g., `specialists/designer.md`)
or by adding ### headings under this section.
