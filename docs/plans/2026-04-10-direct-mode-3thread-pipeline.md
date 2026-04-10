# 3-Thread Pipeline for Direct Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `raune_filter.py::run_single_process()` from single-threaded serial loop to a 3-thread producer-consumer pipeline (decoder → GPU → encoder) with bounded queues, eliminating serialization between stages that run on different hardware units.

**Architecture:** Three Python threads communicating via `queue.Queue(maxsize=2)`. Thread 1 demuxes/decodes via PyAV. Thread 2 calls `_process_batch()` (RAUNE + OKLab on GPU). Thread 3 encodes via PyAV. `None` sentinel propagates termination. Errors captured in shared list with lock + stop_event.

**Tech Stack:** Python 3.13, PyAV, PyTorch + Triton (CUDA), stdlib `threading` + `queue`.

**Spec:** `docs/decisions/2026-04-10-direct-mode-3thread-pipeline.md`

**Issue:** chunzhe10/dorea#64

---

## File Map

| File | Change |
|------|--------|
| `python/dorea_inference/raune_filter.py` | Refactor `run_single_process()` to use 3 threads + bounded queues. `_process_batch()`, `run_pipe_mode()`, OKLab functions, Triton kernel — all UNCHANGED. |

No new files. No new dependencies (`threading` and `queue` are stdlib).

---

## Task 1: Implement 3-thread pipeline in run_single_process

**Files:**
- Modify: `python/dorea_inference/raune_filter.py`

The change is contained entirely within the `run_single_process()` function. The PyAV setup (input/output containers, codec configuration) is preserved unchanged. The serial loop is replaced with three thread definitions, queue creation, thread spawn, and thread join + error propagation.

- [ ] **Step 1: Add stdlib imports at top of file**

Add to the imports section near the top of `raune_filter.py` (after `import time`):

```python
import threading
import queue
```

- [ ] **Step 2: Replace the body of `run_single_process()` after PyAV setup**

Locate the existing `run_single_process()` function. Keep everything from `import av` through the print statement at line ~249 ("[raune-filter] single-process: ...") UNCHANGED. Replace the body from `frame_count = 0` through the end of the function with the new 3-thread implementation below.

The new body (replace from `frame_count = 0` to the end of `run_single_process()`):

```python
    # ─── 3-thread pipeline: decoder → GPU → encoder ────────────────────────
    # Bounded queues provide backpressure (memory bound: ~200MB at 4K)
    q_decoded = queue.Queue(maxsize=2)   # holds: list[np.ndarray] (one batch)
    q_processed = queue.Queue(maxsize=2) # holds: list[np.ndarray] (one batch)

    # Shared error state
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    stop_event = threading.Event()

    def record_error(exc: BaseException) -> None:
        with errors_lock:
            errors.append(exc)
        stop_event.set()

    # Frame counter shared with encoder thread for progress reporting
    t_start = time.time()
    encoded_count = 0

    # ─── Thread 1: Decoder ─────────────────────────────────────────────────
    def decoder_thread() -> None:
        try:
            batch: list[np.ndarray] = []
            for packet in in_container.demux(in_stream):
                if stop_event.is_set():
                    return
                for frame in packet.decode():
                    if stop_event.is_set():
                        return
                    rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                    if rgb.shape[1] != fw or rgb.shape[0] != fh:
                        rgb = np.array(
                            frame.to_image().resize((fw, fh)),
                            dtype=np.uint8,
                        )
                    batch.append(rgb)
                    if len(batch) >= batch_size:
                        q_decoded.put(batch)
                        batch = []
            if batch:
                q_decoded.put(batch)
        except BaseException as e:
            record_error(e)
        finally:
            q_decoded.put(None)  # sentinel: tell GPU thread we're done

    # ─── Thread 2: GPU processing ──────────────────────────────────────────
    def gpu_thread() -> None:
        try:
            while True:
                if stop_event.is_set():
                    return
                batch = q_decoded.get()
                if batch is None:
                    return
                results = _process_batch(
                    batch, model, normalize,
                    fw, fh, pw, ph, transfer_fn,
                )
                q_processed.put(results)
        except BaseException as e:
            record_error(e)
        finally:
            q_processed.put(None)  # sentinel: tell encoder we're done

    # ─── Thread 3: Encoder ─────────────────────────────────────────────────
    def encoder_thread() -> None:
        nonlocal encoded_count
        try:
            while True:
                if stop_event.is_set():
                    return
                results = q_processed.get()
                if results is None:
                    return
                for result_np in results:
                    out_frame = av.VideoFrame.from_ndarray(result_np, format="rgb24")
                    out_frame.pts = encoded_count
                    for pkt in out_stream.encode(out_frame):
                        out_container.mux(pkt)
                    encoded_count += 1
                    if encoded_count % (batch_size * 4) == 0:
                        elapsed = time.time() - t_start
                        fps_actual = encoded_count / elapsed if elapsed > 0 else 0
                        pct = (encoded_count / total_frames * 100
                               if total_frames else 0)
                        print(f"[raune-filter] {encoded_count} frames "
                              f"({pct:.0f}%, {fps_actual:.1f} fps)",
                              file=sys.stderr, flush=True)
        except BaseException as e:
            record_error(e)

    # Spawn threads
    t_dec = threading.Thread(target=decoder_thread, name="decoder", daemon=False)
    t_gpu = threading.Thread(target=gpu_thread, name="gpu", daemon=False)
    t_enc = threading.Thread(target=encoder_thread, name="encoder", daemon=False)

    t_dec.start()
    t_gpu.start()
    t_enc.start()

    # Wait for all threads to finish
    t_dec.join()
    t_gpu.join()
    t_enc.join()

    # Flush encoder (must happen on main thread after encoder thread finishes)
    if not errors:
        for pkt in out_stream.encode():
            out_container.mux(pkt)

    out_container.close()
    in_container.close()

    # Propagate any error from worker threads
    if errors:
        raise errors[0]

    elapsed = time.time() - t_start
    fps_actual = encoded_count / elapsed if elapsed > 0 else 0
    print(f"[raune-filter] done: {encoded_count} frames in {elapsed:.1f}s "
          f"({fps_actual:.2f} fps)",
          file=sys.stderr, flush=True)
    return encoded_count
```

Note: the function returns `encoded_count` (consistent with the existing return value at the end of the old implementation). The variable was previously named `frame_count`; we renamed it to `encoded_count` for clarity since the encoder thread is now the source of truth.

- [ ] **Step 3: Verify the rest of the file is untouched**

After the edit, confirm these are unchanged:
- `_process_batch()` function — same signature, same logic
- `run_pipe_mode()` function — untouched
- `triton_oklab_transfer()`, `pytorch_oklab_transfer()`, OKLab math functions — untouched
- `main()` function — untouched
- All imports near the top other than the two new stdlib imports

Run a syntax check:
```bash
cd /workspaces/dorea-workspace/repos/dorea
python3 -c "import ast; ast.parse(open('python/dorea_inference/raune_filter.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run the filter on the test clip**

```bash
cd /workspaces/dorea-workspace/repos/dorea
./target/release/dorea grade \
  --input "/workspaces/dorea-workspace/footage/raw/2025-11-01/DJI_20251101111428_0055_D_3s.MP4" \
  --output "/workspaces/dorea-workspace/working/oklab_3thread.mov" \
  --output-codec prores \
  --direct \
  --verbose \
  2>&1 | tail -20
```

Expected output:
- Filter prints `[raune-filter] Using Triton fused kernel`
- Filter prints `[raune-filter] single-process: 3840x2160, ...`
- Progress lines like `[raune-filter] 64 frames (18%, X.X fps)` where X.X is significantly higher than 2.4 fps (target ~7 fps)
- `[raune-filter] done: 360 frames in YYY.Ys (Z.ZZ fps)` where Z.ZZ ≥ 5.0
- No traceback, exit code 0
- Output file `oklab_3thread.mov` exists, ~1.3GB, 3.003s duration

If the throughput is below 5 fps, the pipeline isn't actually parallel — investigate GIL contention or stage imbalance before proceeding.

- [ ] **Step 5: Frame-by-frame visual parity check**

Compare a frame from the new 3-thread output against the existing single-threaded output:

```bash
ffmpeg -v quiet -ss 1.5 -i "/workspaces/dorea-workspace/working/oklab_3thread.mov" \
  -frames:v 1 "/workspaces/dorea-workspace/working/oklab_3thread_frame.png" -y

ffmpeg -v quiet -ss 1.5 -i "/workspaces/dorea-workspace/working/oklab_single_process.mov" \
  -frames:v 1 "/workspaces/dorea-workspace/working/oklab_single_process_frame.png" -y

# Compute pixel-level diff
ffmpeg -v error \
  -i /workspaces/dorea-workspace/working/oklab_3thread_frame.png \
  -i /workspaces/dorea-workspace/working/oklab_single_process_frame.png \
  -filter_complex "blend=all_mode=difference,signalstats" \
  -f null - 2>&1 | grep -E "YAVG|YMAX"
```

Expected: YAVG very small (under 1.0), YMAX small (under 5). Identical pipelines should produce identical pixels — any nonzero diff is a bug (frame ordering, race condition).

- [ ] **Step 6: Commit**

```bash
cd /workspaces/dorea-workspace/repos/dorea
git add python/dorea_inference/raune_filter.py
git commit -m "$(cat <<'EOF'
perf(direct): 3-thread pipeline for parallel decode/GPU/encode

Refactor run_single_process() from serial loop to producer-consumer
with three threads and bounded queues. Decoder, GPU processing, and
encoder now run on separate threads, hiding stage latency behind
each other.

PyAV and torch.cuda both release the GIL, so Python threads work
fine here. Queue capacity bounded to 2 batches each (~200MB at 4K)
for backpressure.

Closes #64

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the Implementer

### Why bounded queues
4K frames are 25MB each. An unbounded queue could OOM if one stage stalls (e.g., RAUNE thread temporarily slow). Bounded queues provide backpressure: fast stages block when downstream is full, naturally throttling.

### Why None sentinel and not exceptions
Sentinel-based termination is the standard producer-consumer idiom in Python. It's simpler than exception-based shutdown and doesn't require special handling for clean termination vs error termination — the `errors` list + `stop_event` covers errors separately.

### Why daemon=False
We want clean shutdown. If main thread exits while daemon threads are running, encoded video could be truncated. `daemon=False` plus explicit `join()` ensures all work completes before main returns.

### CUDA thread safety
PyTorch's CUDA context is process-wide and shared across threads. The `_process_batch()` function only runs from Thread 2 (GPU thread) — there's no concurrent CUDA access from multiple threads. This is the correct way to use CUDA from Python threads.

### What if `_process_batch()` blocks indefinitely
If RAUNE inference hangs (e.g., GPU OOM, model crash), the GPU thread blocks in `_process_batch()`. The decoder thread will eventually fill `q_decoded` and block on `put()`. The encoder thread will block on empty `q_processed`. Main thread blocks on `t_dec.join()`. This is a deadlock — but no worse than the existing single-threaded code, which would also hang on a stuck GPU call. CTRL+C from terminal will still terminate the process.

### Don't add features
The plan only changes the threading model. Do NOT:
- Modify `_process_batch()` (that's a separate optimization for a different issue)
- Add the 16-bit precision fix (option B from the deep dive — separate issue)
- Cache full-res tensors on GPU (option C — separate issue)
- Add NVENC support (option D — separate issue)

Keep the change focused on threading.
