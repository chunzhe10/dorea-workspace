# RAG Modes: search vs context vs ask

> Captured 2026-03-10. Updated 2026-03-16.

## Overview

The RAG pipeline (`corvia-kernel/src/rag_pipeline.rs`) has three modes with
increasing cost and capability:

| | search | context | ask |
|---|---|---|---|
| **Stage 1: Retrieval** | Vector search | Vector search + optional graph expansion | Vector search + optional graph expansion |
| **Stage 2: Augmentation** | No | Yes — assembles sources into structured context with token budgets, skill injection | Same as context |
| **Stage 3: Generation** | No | No | Yes — sends assembled context to LLM for a synthesized answer |
| **Returns** | Raw matching entries with scores | Assembled context document + sources + trace | Context + LLM-generated answer + trace |
| **Requires LLM** | No | No | Yes (GenerationEngine) |

## Summary

- **`search`** — raw entries matching the query (cheapest, fastest)
- **`context`** — structured context assembled from retrieved entries (no LLM needed)
- **`ask`** — full RAG: context assembly + LLM-generated answer (requires GenerationEngine)

## RAM implications

- `search` and `context` only need the embedding model (for vector similarity)
- `ask` additionally needs a chat/generation model loaded, which is the main RAM cost
- The GenerationEngine is optional in `RagPipeline::new()` — passing `None` disables
  `ask()` gracefully (returns config error) while `context()` and search still work

## Resolution

The `generation_enabled` config flag was not added as a separate toggle. Instead,
the inference architecture naturally supports this: corvia-inference loads embedding
and chat models independently. If no chat model is configured in `[inference.chat_models]`,
the server starts with embedding-only capability. The `ask()` pipeline returns a config
error when no GenerationEngine is available, while `search()` and `context()` work
normally. This makes the single-flag approach unnecessary — the model configuration
itself is the toggle.
