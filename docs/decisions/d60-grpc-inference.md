# D60: gRPC Inference Server (corvia-inference)

**Date:** 2026-03-02
**Status:** Shipped (v0.3.1)
**Supersedes:** D40 (Ollama as default inference engine)

## Decision

Custom gRPC inference server (`corvia-inference`) using ONNX Runtime replaces Ollama as
the default embedding provider. The server handles both embedding (via ONNX Runtime /
fastembed) and chat inference (via candle / llama-cpp-2), communicated over protobuf on
HTTP/2. Ollama remains available as an opt-in alternative backend.

## Context

Ollama adds ~500MB binary size and requires separate installation, contradicting the
zero-Docker, zero-dependency promise of LiteStore (D35). For embeddings (a single forward
pass), most of Ollama's LLM-serving features are dead weight. The HTTP/JSON layer adds
network latency and 2x serialization overhead for float vectors -- a 768-dim f32 vector
is 3,072 bytes in packed protobuf vs ~6,000 bytes in JSON. Direct ONNX Runtime also
unlocks hardware-specific execution providers (CoreML, OpenVINO, CUDA, TensorRT) that
Ollama's llama.cpp does not support for encoder models.

## Consequences

- Two new crates: `corvia-proto` (shared protobuf definitions) and `corvia-inference`
  (gRPC server binary).
- Three proto services: `EmbeddingService` (Embed, EmbedBatch, ModelInfo),
  `ChatService` (Chat, ChatStream), `ModelManager` (ListModels, LoadModel, Health).
- Default port 8030 with `nomic-embed-text-v1.5` (768 dimensions) as default embedding
  model. Models auto-downloaded from HuggingFace Hub on first use.
- `InferenceProvider` enum gains `Corvia` as the new default variant alongside `Ollama`
  and `Vllm`.
- `GrpcInferenceEngine` implements `InferenceEngine` trait; `GrpcChatEngine` implements
  `ChatEngine` trait. MergeWorker now takes `Arc<dyn ChatEngine>` instead of a raw
  Ollama URL.
- Existing tests unaffected -- all use `MockEngine`. Real model tests gated behind
  `--features real-inference`.

**Source:** `repos/corvia/docs/rfcs/2026-03-02-grpc-inference-server-design.md`
