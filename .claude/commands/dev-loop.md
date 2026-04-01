Use the dev-loop skill from `.agents/skills/dev-loop/SKILL.md` to execute the full
development lifecycle for the given GitHub issue.

**Issue:** $ARGUMENTS

## Instructions

Read and follow `.agents/skills/dev-loop/SKILL.md` exactly. The skill defines 7 phases:

1. **Issue Intake** — Fetch the issue with `gh issue view`, query corvia for context, create feature branch
2. **Brainstorm & Design** — Invoke `superpowers:brainstorming` with the issue prompt
3. **Implementation Plan** — Invoke `superpowers:writing-plans` from the approved design
4. **Implementation** — Invoke `superpowers:subagent-driven-development` to execute the plan
5. **5-Persona Review** — Dispatch 5 independent reviewer subagents (Senior SWE, PM, QA + 2 dynamic) using `.agents/skills/dev-loop/five-persona-reviewer.md` template. Loop fixes until all Critical/Important/Low issues resolved.
6. **E2E Integration Test** — Test as a real user using `.agents/skills/dev-loop/e2e-tester.md`. Cover happy path, edge cases, integration points, regression.
7. **Branch, PR, Merge** — Push, create PR with review summary, merge to master, resolve conflicts if any, cleanup branch, record in corvia.

**Hard gates:** No phase can be skipped. Design must be approved before planning. All review issues (Critical through Low) must be fixed before merge. Tests must pass after conflict resolution.

**Autonomy:** Proceed autonomously through all phases. Only pause to ask if the issue is ambiguous, brainstorming produces fundamentally different approaches, or merge conflicts are complex.

Start now by announcing: "I'm using the dev-loop skill to work on issue #$ARGUMENTS."
