# GPU Inference: Device & Backend Control

**Date:** 2026-03-11
**Status:** Shipped (v0.4.2)
**Supersedes:** None (extends D60 gRPC inference server)

## Decision

OpenVINO for Intel iGPU embedding, CUDA for NVIDIA GPU, with runtime backend switching
via config. The `LoadModelRequest` gRPC message accepts `device` (auto/gpu/cpu) and
`backend` (cuda/openvino/vulkan/cpu) fields. Responses report `actual_device` and
`actual_backend`, implementing a fallback-with-signal pattern: if GPU is unavailable,
the model loads on CPU successfully but the caller sees the mismatch.

## Context

CPU-only inference is slow for embedding workloads. Users have diverse GPU hardware --
Intel integrated GPUs (common in laptops and devcontainers via WSL) and NVIDIA discrete
GPUs. The corvia-inference server needed a way to select the appropriate execution
provider at runtime without recompilation, while gracefully degrading when requested
hardware is unavailable.

## Consequences

- `backend.rs` module with `resolve_backend()` function, `ResolvedBackend` struct
  (device, backend kind, fallback flag), and GPU availability probing cached at startup.
- Embedding service switched from `fastembed::InitOptions` to `InitOptionsUserDefined`
  to unlock per-model execution provider control. Model registry maps friendly names to
  `UserDefinedEmbeddingModel` structs (ONNX paths, tokenizer config, dimensions).
- Chat service configures `LlamaModelParams` GPU layer offloading based on resolved
  backend. OpenVINO rejected for chat models (llama.cpp does not support it).
- `[inference]` config section with `device`, `backend`, and `embedding_backend` fields.
  Default `device = "auto"` prefers NVIDIA/CUDA when both NVIDIA and Intel GPUs are
  present. Intel GPU used only when explicitly requested via `backend: "openvino"`.
- Cargo features: `ort` pinned with `openvino` + `cuda` features, `fastembed` with
  `cuda` feature, `llama-cpp-2` with `cuda` feature.
- Devcontainer: NVIDIA GPU works via `--gpus all`. Intel GPU passthrough documented but
  opt-in (commented out `--device=/dev/dri` in `devcontainer.json`). Dockerfile includes
  Intel OpenCL/Level Zero runtime packages.
- Out of scope: NPU support, hot-switching (requires unload + reload), Vulkan for
  llama-cpp-2 (deferred).

**Source:** `repos/corvia/docs/rfcs/2026-03-11-inference-gpu-device-control-design.md`
