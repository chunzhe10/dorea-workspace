# Dev-Loop Issue Template

Use this template when creating GitHub issues intended for autonomous dev-loop
execution. Each section maps to a dev-loop phase, minimizing intake time and
maximizing implementation quality.

---

## Template

```markdown
## Context

[1-3 sentences: what is the broader feature/initiative this issue belongs to.
Link to RFC or design doc. Link to parent issue if this is part of a series.]

## Goal

[1-2 sentences: what this specific issue delivers. Be precise about the outcome,
not the process.]

## Design Reference

[Link to the RFC section(s) that specify this work. Quote the key decisions if
short. The dev-loop brainstorm phase can skip exploration if the design is already
approved.]

## Prompt

[The instruction an autonomous agent should follow. Write this as if you are
giving a clear task to a senior developer who has access to the codebase but no
prior context beyond what is in this issue and the linked docs.

Include:
- What to build (files to create/modify)
- Key constraints (patterns to follow, traits to implement, protocols to respect)
- What NOT to do (scope boundaries)
- How to verify (specific test commands or scenarios)]

## Acceptance Criteria

[Checklist of observable outcomes. Each item should be verifiable by running a
command or reading a file. These become the pass/fail criteria for Phase 5 review
and Phase 6 E2E testing.]

- [ ] Criterion 1 (e.g., "hook script validates team name against regex")
- [ ] Criterion 2 (e.g., "staging directory created with 0700 permissions")
- [ ] Criterion N

## Labels

[Choose labels that drive dynamic reviewer selection in the 5-persona review:]

| Label | Dynamic Reviewers Added |
|-------|----------------------|
| `enhancement` | Domain Expert + Developer Experience |
| `security` | Security Engineer + Compliance Reviewer |
| `performance` | Performance Engineer + Storage Specialist |
| `infrastructure` | SRE/Platform Engineer + Container Specialist |
| `api` | API Design Reviewer + Backwards Compat Reviewer |
| `rust` | Rust Idiom Reviewer + Unsafe/Lifetime Reviewer |

## Dependencies

[Issues that must be completed before this one. Use `Depends on #N` syntax.
If none, say "None -- can start immediately."]

## Files Likely Changed

[List the files the dev-loop agent will probably need to read and modify. This
speeds up Phase 1 context gathering.]
```

---

## Title Convention

Issue titles are plain descriptive text. No conventional-commit prefixes
(`feat:`, `fix:`, etc.) -- those are for git commits, not issues.

- Single issue: `Descriptive title of the work`
- Multi-phase: `Topic Phase N: description` (e.g., `BM25 Hybrid Search Phase 2a: tantivy integration`)
- Tracking issue: `Topic: tracking issue`

## Guidelines

**One dev-loop run per issue.** Each issue should be completable in a single
autonomous session (typically 30-90 minutes). If an issue would take multiple
sessions, split it.

**Design-complete issues skip brainstorming.** If the issue links to an approved
RFC section with sufficient detail, the dev-loop agent can skip Phase 2
(brainstorm) and go directly to Phase 3 (implementation plan). Signal this in
the Prompt section: "Design is approved in RFC section N. Skip brainstorming,
proceed to implementation planning."

**Acceptance criteria are the contract.** The 5-persona review and E2E testing
use these as pass/fail gates. Vague criteria ("works correctly") produce vague
reviews. Specific criteria ("T3 test passes: concurrent flock writes produce
valid JSONL") produce actionable reviews.

**Labels matter.** They determine which dynamic reviewers are selected. An issue
labeled `security` gets Security Engineer + Compliance Reviewer. Without the
label, those reviewers are not dispatched, and security issues may be missed.

**Prompt is for the agent, not the human.** Write it assuming the reader has no
memory of prior conversations. Include enough context that the agent can work
autonomously. Reference specific files, functions, and patterns by name.
