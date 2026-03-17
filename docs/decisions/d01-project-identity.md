# D01: Project Identity -- Corvia

**Date:** 2026-02-25
**Status:** Shipped (v0.1.0)
**Supersedes:** None

## Decision

Named the project "Corvia" (derived from Corvus, the raven constellation), positioned
as a knowledge-gathering system for AI agents, and licensed under AGPL-3.0. The name
references Odin's ravens Huginn and Muninn who gathered knowledge daily -- mapping
directly to the concept of agents collecting and returning organizational knowledge.
The all-Rust tech stack was chosen for performance (4,700 QPS MCP servers vs Python's
~300 QPS) and portfolio differentiation over the LangChain/LangGraph ecosystem.

## Context

The project needed a distinctive name and positioning for a knowledge layer that goes
beyond individual agent memory. Alternatives considered included Lyra (harmony metaphor,
too generic), Corvus (too dark/Gothic), and Munin (strong mythology but harder to
spell internationally). The broader identity decisions -- AGPL-3.0 licensing, all-Rust
implementation, and CLI-first form factor -- were shaped by the goal of building a
genuinely useful open-source tool with commercial optionality, rather than a portfolio
showcase alone.

## Consequences

- All-Rust monorepo (`repos/corvia/`) as a Cargo workspace with 9+ crates.
- AGPL-3.0 license provides SaaS protection while remaining fully OSI-approved open
  source. Dual-licensing path planned for enterprises with blanket AGPL bans.
- Python excluded from production code (reserved only for fine-tuning experiments).
- CLI is the primary interface; VS Code extension is a demo/showcase frontend.
- Self-documenting (dogfooding) from day one -- Corvia's own knowledge base is built
  and queried using Corvia itself.

**Source:** `repos/corvia/docs/rfcs/2026-02-25-corvia-brainstorm.md`
