# 3-Thread Pipeline for Direct Mode

## Problem

Direct mode runs at 2.35 fps on 4K 120fps HEVC. After eliminating pipe I/O, the bottleneck
is **lack of pipelining** — decode, GPU processing, and encode in `raune_filter.py::run_single_process()`
run serially in a single thread:

```python
for packet in demux:
    for frame in decode:           # CPU: blocks GPU + encode
        batch.append(frame)
        if batch full:
            process_batch()         # GPU: blocks decode + encode
            for r in results:
                encode(r)           # CPU: blocks decode + GPU
```

These three stages run on different hardware units (CPU decoder, GPU, CPU encoder) but
cannot overlap. Theoretical pipelined ceiling is ~12.5 fps; we waste ~200-300ms per frame
on serialization.

## Decision

Producer-consumer architecture with three threads and bounded queues:

```
Thread 1 (decoder):  PyAV decode → q_decoded   (capacity 2 batches)
Thread 2 (GPU):      q_decoded → process → q_processed   (capacity 2 batches)
Thread 3 (encoder):  q_processed → PyAV encode
```

### Why threads, not processes
- PyAV releases the GIL during decode/encode (libav calls)
- `torch.cuda` releases the GIL during CUDA operations
- numpy operations release the GIL for large arrays
- Shared memory — no IPC serialization overhead
- The only GIL-bound work is short Python orchestration

### Queue sizing
- Capacity = 2 batches each (8 frames at batch_size=4)
- Memory bound: ~200MB (8 × 25MB at 4K uint8)
- Large enough to hide stage variability, small enough to bound memory

### Termination
A `None` sentinel propagates downstream:
1. Decoder finishes → puts `None` on `q_decoded`
2. GPU thread drains, puts `None` on `q_processed`
3. Encoder drains, exits

### Error handling
- Each worker thread catches exceptions, appends to shared `errors` list (lock-protected)
- A `stop_event: threading.Event` signals other threads to abort
- Sentinel `None` ensures downstream queues unblock
- Main thread joins all threads then raises any captured exception

### CUDA thread safety
- PyTorch CUDA context is process-wide; calling from a single Python thread is fine
- Only Thread 2 touches CUDA — no contention
- Pinned memory could be added later for additional async copy benefits

## Expected Outcome

- Slowest stage (~80ms RAUNE or ~80ms encode) determines throughput
- Target: 2.35 fps → ~7 fps (3× speedup)

## Scope

- Modify only `raune_filter.py::run_single_process()`
- Pipe mode (`run_pipe_mode`) untouched
- `_process_batch()` unchanged — same signature, same logic
- No Rust changes
- No new dependencies (Python `threading` and `queue` are stdlib)

## Alternatives Rejected

### Multiprocessing (separate processes per stage)
- Higher overhead — IPC serialization of frame data
- More memory usage — no shared GPU context
- Threads work fine because PyAV/torch.cuda release GIL

### asyncio
- Decode/encode are not natively async; would need to wrap in executors
- More complex than threading.Thread + queue.Queue
- No benefit over threads for this use case

### Rust-side parallelism
- Would require larger refactor (Rust orchestrating Python subprocess pool)
- Direct mode design intentionally puts orchestration in Python
- Out of scope for this issue

## Spec / Plan

- Spec: this document
- Plan: `docs/plans/2026-04-10-direct-mode-3thread-pipeline.md`
- Issue: chunzhe10/dorea#64
