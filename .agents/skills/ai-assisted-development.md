# AI-Assisted Development Patterns

Curated patterns from community best practices (including
[everything-claude-code](https://github.com/affaan-m/everything-claude-code), Anthropic
hackathon winner) adapted for any AI coding assistant. Apply these while developing corvia.

## Context Management

**Guard the context window aggressively.**

- Keep active tool count reasonable. Too many MCP tools / plugins can halve your effective context.
- Delegate research and exploration to subagents / background tasks — they use separate context.
- Keep files modular (hundreds of lines, not thousands) — improves both token cost and first-attempt success.
- Start fresh sessions per task. Clear context between unrelated work.
- Compact or summarize proactively at ~70% context usage, at logical breakpoints.

**Model selection (when applicable):**
- Default to the fastest sufficient model for subtasks.
- Upgrade to stronger models when: first attempt fails, task spans 5+ files, or decisions are architecture/security-critical.

## Session Continuity

**Don't lose progress across sessions.**

- Save session state to files before context clears (approaches that worked, failed attempts, remaining work).
- Start new sessions by referencing previous state files or knowledge entries.
- When encountering repeated patterns, capture them in AGENTS.md or knowledge entries — not ad-hoc notes.

**For corvia specifically:** Use `corvia_write` (via MCP) to record design decisions and patterns into the knowledge store. Dogfood the product.

## Verification

**Verification criteria = highest-leverage practice.**

- Give explicit pass/fail criteria before implementing. AI assistants perform dramatically better when they can verify their own work.
- Use checkpoint-based verification: define criteria at each stage, verify before proceeding.
- When uncertain, test multiple approaches and compare results.

## Parallelization

**Minimum viable parallelization — quality over quantity.**

- Git worktrees for isolated parallel work on different branches.
- The Cascade Method: open new tasks right-to-left by age, sweep oldest-to-newest. Max 3-4 concurrent.
- Two-Instance Kickoff: one for scaffolding/structure, one for research/requirements.

## Subagent / Background Task Patterns

**Delegated tasks lack semantic context about WHY a query matters.**

- Use iterative retrieval: orchestrator evaluates returns, asks follow-ups, subagent retrieves again (max 3 cycles).
- Scope delegated tools deliberately — fewer tools = more focused execution.
- Sequential phases produce files that feed next phase: Research -> Plan -> Implement -> Review -> Verify.

## AGENTS.md as Infrastructure

**Context engineering > prompt engineering.**

- AGENTS.md is as important as .gitignore — essential infrastructure, not optional documentation.
- Include: build commands, code style rules, architecture decisions, gotchas the AI can't infer from code alone.
- Keep modular: split by concern when it grows large (security, testing, workflow).
- Update continuously as patterns stabilize. Don't let it drift from reality.

## Automation Hooks

**Automate what you'd otherwise forget.**

Most AI coding tools support some form of lifecycle hooks or automation triggers:

| Hook Type | Purpose | Example |
|-----------|---------|---------|
| Pre-tool | Validate before execution | Prevent dangerous commands |
| Post-tool | Auto-format, lint after edits | Run formatter on saved files |
| Session end | Persist learnings | Save session state to file |
| Pre-compact | Save important state | Checkpoint before context shrinks |

## Skills / Instructions Management

- Limit active skills/instructions to 20-30 high-quality, task-specific ones.
- Skills = reusable techniques. Project-specific conventions go in AGENTS.md.
- One excellent example beats many mediocre ones.
- Store skills in the project repo (`.agents/skills/`) so all team members and tools benefit.

## Anti-Patterns to Avoid

- Over-configuring before you need it — configuration is fine-tuning, not architecture.
- Wrapping CLIs in MCP tools when a simpler reference doc would preserve more context.
- Running parallel instances with overlapping code changes without worktree isolation.
- Skipping verification because "it looks right."
- Keeping context from unrelated tasks — fresh sessions are your friend.

## Applying to corvia Development

1. **Start tasks with corvia MCP** — `corvia_search` or `corvia_ask` for prior decisions before touching code.
2. **Record decisions back** — `corvia_write` to persist what you learn for future sessions.
3. **Verify with tier-1 tests** — `cargo test --workspace` before claiming anything works.
4. **Keep AGENTS.md current** — update as architecture evolves.
5. **Delegate exploration** — use subagents for broad codebase questions, direct tools for specific lookups.

## Sources

- [everything-claude-code](https://github.com/affaan-m/everything-claude-code) — Anthropic hackathon winner
- [the-shortform-guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-shortform-guide.md) — setup & foundations
- [the-longform-guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-longform-guide.md) — optimization, memory, evals, parallelization
- [AGENTS.md standard](https://agents.md/) — cross-platform AI agent instructions
