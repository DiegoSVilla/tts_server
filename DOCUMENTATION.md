# XTTS-v2 TTS Server - Technical Documentation

## High-Level Overview

This document provides comprehensive technical documentation for developers and AI agents working with the XTTS-v2 Streaming TTS Server.

### Architecture Summary

The server is a FastAPI-based WebSocket application that streams real-time audio from text using Coqui's XTTS-v2 model. Key architectural decisions:

1. **Model Loading**: XTTS-v2 loaded directly from `models/XTTS-v2/` directory (HuggingFace download), bypassing TTS.api wrapper for full parameter control
2. **Speaker Conditioning**: Pre-computed speaker latents on startup for languages "en" and "pt" using `test_speaker_shrt.wav`
3. **Streaming Implementation**: Uses `inference_stream()` with basic parameters only (text, language, gpt_cond_latent, speaker_embedding)
4. **Event Loop Management**: Inference runs in thread pool executor (`run_in_executor`) to avoid blocking async event loop
5. **GPU Selection**: Auto-selects GPU with most free memory by default using `select_best_gpu()`

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Server | `main.py` | FastAPI app with WebSocket endpoint, GPU management, model lifecycle |
| Tests | `main_test.py` | Complete test suite (unit + integration + E2E) |
| Launcher | `start.bat` | Windows launcher with optional test mode |
| Model | `models/XTTS-v2/` | XTTS-v2 checkpoint files from HuggingFace |

### Data Flow

```
Client                          Server
  │                               │
  │── JSON: {text, language} ───▶│
  │                               │
  │                               │── Get cached speaker latents
  │                               │── Run inference in executor
  │                               │── Stream audio chunks
  │◀── Binary: audio data ────────│
  │                               │
  │── JSON: {end: true} ─────────▶│
  │                               │
  │◀── Text: "DONE" ──────────────│
```

---

## Server Implementation

### Global Configuration

The server uses module-level constants for runtime configuration:

```python
SELECTED_DEVICE: str = "cpu"        # Selected GPU/CPU device
USE_FP16: bool = False              # FP16 quantization enabled
SPEED_MULTIPLIER: float = 1.0       # Speech speed multiplier
TEMPERATURE: float = 0.75           # Sampling temperature
LENGTH_PENALTY: float = 1.0         # Output length control
REPETITION_PENALTY: float = 10.0    # Repetition prevention
TOP_K: int = 50                     # Top-k sampling
TOP_P: float = 0.85                 # Top-p (nucleus) sampling
MODEL_DIR: str = "models/XTTS-v2"   # Model checkpoint directory
```

### GPU Management Functions

#### `get_gpu_stats() -> list[dict]`

Returns memory statistics for all CUDA devices.

**Output format:**
```python
[
    {
        "device": 0,
        "name": "NVIDIA GeForce RTX 2080 Ti",
        "total_gb": 23.62,
        "free_gb": 23.62,
        "allocated_gb": 0.0,
        "utilization_pct": 0.0
    }
]
```

**Behavior:**
- Returns empty list if CUDA unavailable
- Pure function with no side effects
- ~1ms execution time per GPU

#### `select_best_gpu() -> Optional[int]`

Selects GPU with most free memory.

**Algorithm:**
1. Call `get_gpu_stats()`
2. Find GPU with maximum `free_gb`
3. Log selection: `f"Auto-select: GPU {idx} has {free}GB free (best)"`
4. Return device index or None

#### `log_gpu_stats(selected_device: str) -> None`

Logs formatted GPU statistics to console with visual usage bars.

**Example output:**
```
GPU  0: NVIDIA GeForce RTX 2080 Ti     | ########----------------------------------------|    8.50GB /  23.62GB ( 35.0%) <- SELECTED
GPU  1: Tesla P40                      | ------------------------------------------------|    3.95GB /  24.05GB ( 16.4%)
```

### Argument Parsing

#### `get_args() -> argparse.Namespace`

Parses command-line arguments with the following precedence:

1. `--auto-select-gpu` takes precedence if provided
2. Else use `--cuda-device` if specified
3. Default to auto-select (changed from cuda:0)
4. Fall back to CPU if no CUDA

**Arguments:**
- `--cuda-device N`: Force specific GPU (0, 1, 2, ...)
- `--auto-select-gpu`: Auto-select GPU with most free memory
- `--port`: Server port (default: 8002)
- `--host`: Server host (default: 0.0.0.0)
- `--fp16`: Enable FP16 quantization
- `--speed N`: Speech speed multiplier (default: 1.0)
- `--temperature N`: Sampling temperature (default: 0.75)
- `--top-k N`: Top-k sampling (default: 50)
- `--top-p N`: Top-p sampling (default: 0.85)

### Server Lifecycle

#### Startup Sequence

1. Parse command-line arguments via `get_args()`
2. Check CUDA availability with `torch.cuda.is_available()`
3. Select device based on args (auto-select by default)
4. Log all GPU statistics via `log_gpu_stats()`
5. Log optimization settings (FP16, speed, temperature, etc.)
6. Load TTS model on selected device via `manager.load_tts_model()`
7. Pre-compute speaker latents for "en" and "pt" languages
8. Start server

#### Model Loading

The `ConnectionManager.load_tts_model()` method:

1. Loads config from `models/XTTS-v2/config.json`
2. Initializes Xtts model from config
3. Loads checkpoint from `models/XTTS-v2/` directory
4. Moves model to selected device (GPU or CPU)
5. Applies FP16 quantization if enabled (`model.half()`)
6. Sets model to eval mode
7. Loads speaker latents from `test_speaker_shrt.wav` for "en" and "pt"

**Speaker Latents:**
- Computed once on startup using `get_conditioning_latents()`
- Cached in `self.speaker_latents[language]` dict
- Returns tuple: `(gpt_cond_latent, speaker_embedding)`
- Used for all inference requests (no per-request speaker wav)

### WebSocket Protocol

#### Client Message Format

```json
{
  "text": "Text to convert to speech",
  "language": "en",
  "end": false
}
```

**Fields:**
- `text` (required): Text to synthesize
- `language` (optional, default "en"): Language code
- `end` (optional, default false): Set true for last chunk

**Note:** `speaker_wav` parameter removed - uses pre-computed latents only

#### Server Response Format

**Success:**
- Binary frames: Raw audio data (24kHz, 16-bit PCM, mono)
- Text frame: `"DONE"` when `end: true` received

**Error:**
- Text frame: `"ERROR: <error message>"`

#### Streaming Inference

The `stream_inference()` method:

1. Gets cached speaker latents for language
2. Runs `inference_stream()` in executor to avoid blocking
3. Receives generator of audio chunks (torch.Tensor or numpy)
4. Encodes each chunk to 24kHz 16-bit PCM
5. Sends encoded audio via WebSocket immediately

**Inference Parameters (removed in streaming):**
- `inference_stream()` only accepts: text, language, gpt_cond_latent, speaker_embedding
- Optimization parameters (speed, temperature, top_k, top_p) not used in streaming
- These would apply to non-streaming `inference()` method

**Performance:**
- First chunk arrives in ~0.6-0.9s
- Full chunk generation takes 1.5-3.5s depending on text length
- Runs in thread pool to avoid blocking async event loop

---

## API Endpoints

### WebSocket

#### `GET /ws`

Real-time TTS streaming endpoint.

**Protocol:**
1. Client connects via WebSocket
2. Client sends JSON messages with text chunks
3. Server responds with binary audio chunks
4. Client sets `end: true` for last chunk
5. Server sends "DONE" text message

**Audio Format:**
- Sample rate: 24,000 Hz
- Bit depth: 16-bit
- Channels: Mono (1)
- Encoding: Signed little-endian PCM

### HTTP Endpoints

#### `GET /health`

Returns server status and device information.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_connections": 1,
  "device": "cuda:0",
  "cuda_available": true,
  "cuda_device_count": 2
}
```

#### `GET /gpu-stats`

Returns detailed GPU memory statistics.

**Response:**
```json
{
  "selected_device": "cuda:1",
  "cuda_available": true,
  "device_count": 2,
  "devices": [
    {
      "device": 0,
      "name": "NVIDIA RTX 3080",
      "total_gb": 10.0,
      "free_gb": 1.2,
      "allocated_gb": 8.5,
      "utilization_pct": 85.0
    }
  ]
}
```

#### `GET /languages`

Returns supported language codes.

**Response:**
```json
{
  "languages": [
    "en", "pt", "es", "fr", "de", "it",
    "pl", "ru", "nl", "cs", "tr", "ar",
    "zh", "ja", "ko", "hi"
  ]
}
```

---

## Test Suite

### Structure

The test suite in `main_test.py` is organized into sections:

1. **Unit Tests - GPU Management** (run without server)
   - GPU Statistics gathering
   - Auto GPU Selection algorithm
   - GPU Logging output

2. **Unit Tests - Argument Parsing** (run without server)
   - Command-line argument validation

3. **Integration Tests - WebSocket** (requires running server)
   - Basic WebSocket connection
   - Simple inference test

4. **E2E Tests - Streaming** (requires running server)
   - Spirit story streaming (18 chunks)
   - Full audio generation and validation

### Test Execution

**Run all tests:**
```bash
python main_test.py
```

**Tests require server running at ws://localhost:8002/ws for integration/E2E tests**

### Test Coverage

| Functionality | Test Type | Covered |
|---------------|-----------|---------|
| GPU stats | Unit | ✅ |
| Auto GPU selection | Unit | ✅ |
| GPU logging | Unit | ✅ |
| Argument parsing | Unit | ✅ |
| WebSocket connection | Integration | ✅ |
| Streaming inference | E2E | ✅ |
| Audio generation | E2E | ✅ |

---

## Performance Benchmarks

### Placeholder for Future Benchmarks

This section is reserved for building comprehensive performance benchmarks.

**Recommended benchmark metrics:**
- Inference time by text length (50, 150, 300, 500 chars)
- Characters per second
- Audio output size
- Number of chunks generated
- First chunk latency
- Total generation time

**Benchmark configurations to test:**
- FP32 vs FP16
- Different speed multipliers
- Different temperature values
- Different GPU models

**Note:** The old `test_performance.py` file was removed as part of the reorganization. Future benchmarks should be added to `main_test.py` or a new dedicated benchmark module.

---

## File Structure

### Current Files

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | Server implementation | ✅ Keep |
| `main_test.py` | Complete test suite | ✅ Keep |
| `start.bat` | Windows launcher | ✅ Keep |
| `README.md` | User documentation | ✅ Keep |
| `DOCUMENTATION.md` | Technical documentation | ✅ Keep (this file) |

### Files to Delete

| File | Reason |
|------|--------|
| `test_client.py` | Replaced by main_test.py |
| `test_gpu_management.py` | Replaced by main_test.py |
| `test_performance.py` | Benchmarks removed (placeholder in docs) |
| `test_model.py` | Replaced by main_test.py |
| `IMPLEMENTATION_SUMMARY.md` | Merged into DOCUMENTATION.md |
| `OPTIMIZATION_SUMMARY.md` | Merged into DOCUMENTATION.md |
| `test.bat` | Replaced by start.bat test mode |

---

## Key Design Decisions

### 1. Pre-computed Speaker Latents

**Decision:** Compute speaker conditioning latents once on startup, cache for reuse

**Rationale:**
- Faster inference (no per-request latent computation)
- Simplified client protocol (no speaker_wav parameter)
- Limited to pre-configured speakers only

**Trade-off:** Cannot dynamically change speaker per request

### 2. Streaming Inference Only

**Decision:** Use `inference_stream()` instead of `inference()`

**Rationale:**
- Real-time audio streaming capability
- Lower latency (first chunk in ~0.6-0.9s)
- Better user experience for long texts

**Trade-off:** Some optimization parameters not available in streaming mode

### 3. Executor for Inference

**Decision:** Run inference in thread pool executor

**Rationale:**
- Avoids blocking async event loop
- Allows concurrent connections
- Better server responsiveness

**Trade-off:** Slight overhead from thread management

### 4. Auto-select GPU by Default

**Decision:** Changed default from `cuda:0` to auto-select

**Rationale:**
- Works better for multi-GPU systems
- Prevents OOM errors automatically
- Better out-of-the-box experience

**Trade-off:** Less predictable which GPU is used

---

## Security Considerations

**Production deployment requirements:**

1. **Authentication:** Add WebSocket authentication
2. **Rate Limiting:** Implement request rate limiting per client
3. **Input Validation:** Sanitize text inputs
4. **Network Security:** Bind to specific interfaces, use TLS/SSL
5. **Resource Limits:** Set memory/CPU limits to prevent abuse

---

## Future Enhancements

**Potential additions:**

1. Dynamic speaker selection via API
2. Multiple concurrent speakers
3. Benchmark test suite
4. Web dashboard for monitoring
5. Batch processing for efficiency
6. Caching for common phrases
7. Model compilation with `torch.compile()`
8. DeepSpeed for model parallelism

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Check GPU memory: `curl http://localhost:8002/gpu-stats`
- Solution: Use auto-select or different GPU

**2. Model Not Loading**
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Solution: Use CPU fallback or fix CUDA installation

**3. Speaker Latents Missing**
- Ensure `test_speaker_shrt.wav` exists in project root
- Solution: Add speaker audio file or download from HuggingFace

**4. Connection Errors**
- Verify server running: `curl http://localhost:8002/health`
- Check firewall/port conflicts

---

## References

- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [XTTS-v2 Model Card](https://huggingface.co/coqui/XTTS-v2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)