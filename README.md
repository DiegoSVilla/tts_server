# XTTS-v2 Streaming TTS Server

A high-performance WebSocket-based Text-to-Speech server using Coqui's XTTS-v2 model with voice cloning capabilities. Supports real-time streaming, multi-language synthesis, and intelligent GPU management.

## Features

- **🎯 Real-time Streaming**: Send text chunks sequentially, receive audio chunks as they're generated
- **🗣️ Voice Cloning**: Clone any voice with a 6-30 second reference audio sample
- **🌍 Multi-language Support**: 16+ languages (English, Portuguese, Spanish, French, German, Italian, Polish, Russian, Dutch, Czech, Turkish, Arabic, Chinese, Japanese, Korean, Hindi)
- **🎮 GPU Management**: Intelligent GPU selection with memory monitoring and auto-selection
- **⚡ Performance Optimizations**: FP16 quantization, adjustable speed, and sampling parameters for faster inference
- **📡 WebSocket API**: Simple, efficient binary protocol for real-time audio streaming
- **🔧 Developer-Friendly**: Comprehensive API documentation, health checks, and monitoring endpoints

## Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd tts_server

# Create virtual environment and install dependencies (using uv)
uv venv
uv sync

# Alternative: using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Server

**Default (auto-select GPU with most free memory):**
```bash
# Windows
start.bat

# Linux/Mac
uv run python main.py --auto-select-gpu
```

**Use specific GPU:**
```bash
uv run python main.py --cuda-device 0  # Use GPU 0
uv run python main.py --cuda-device 1  # Use GPU 1
```

**Use CPU (if no GPU available):**
```bash
uv run python main.py  # Will use cuda:0 by default if available
```

### Server Configuration

**GPU Selection Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--cuda-device N` | `None` | Force specific CUDA device (0, 1, 2, ...) |
| `--auto-select-gpu` | `False` | Auto-select GPU with most free memory |

**Performance Optimization Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fp16` | `False` | Enable FP16 half-precision (2x faster, lower VRAM) |
| `--speed N` | `1.0` | Speech speed multiplier (1.5 = 50% faster) |
| `--temperature N` | `0.75` | Sampling temperature (0.5-1.0) |
| `--top-k N` | `50` | Top-k sampling parameter |
| `--top-p N` | `0.85` | Top-p (nucleus) sampling parameter |

**General Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `8002` | Server port number |
| `--host` | `0.0.0.0` | Server host address |

**Examples:**
```bash
# Auto-select GPU (recommended for multi-GPU systems)
uv run python main.py --auto-select-gpu

# Force GPU 1 (useful if GPU 0 is busy)
uv run python main.py --cuda-device 1

# Enable FP16 for faster inference (recommended)
uv run python main.py --fp16 --auto-select-gpu

# Increase speech speed by 50%
uv run python main.py --speed 1.5

# Optimal performance: FP16 + faster speech
uv run python main.py --fp16 --speed 1.3 --temperature 0.7

# Change port
uv run python main.py --port 8003

# Run on localhost only
uv run python main.py --host 127.0.0.1
```

## API Reference

### WebSocket Protocol

**Endpoint:** `ws://localhost:8002/ws`

#### Request Format

Send JSON messages with the following structure:

```json
{
  "text": "Text to convert to speech",
  "speaker_wav": "path/to/reference/audio.wav",
  "language": "en",
  "end": false
}
```

**Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize |
| `speaker_wav` | string | Yes* | - | Path to reference audio (6-30s WAV/MP3/M4A) |
| `language` | string | No | `"en"` | Language code (see supported languages below) |
| `end` | boolean | No | `false` | Set to `true` for last chunk (triggers "DONE" response) |

*Required on first message; subsequent messages can omit it

#### Response Format

**Success:**
- **Binary frames**: Raw audio data (24kHz, 16-bit PCM, mono)
- **Text frame**: `"DONE"` when `end: true` is received

**Error:**
- **Text frame**: `"ERROR: <error message>"`

#### Audio Specifications

- **Sample Rate**: 24,000 Hz
- **Bit Depth**: 16-bit
- **Channels**: Mono (1)
- **Format**: Signed little-endian PCM (raw binary)
- **Container**: None (raw audio data)

### HTTP Endpoints

#### Health Check

```http
GET /health
```

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

#### GPU Statistics

```http
GET /gpu-stats
```

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
    },
    {
      "device": 1,
      "name": "NVIDIA RTX 4090",
      "total_gb": 24.0,
      "free_gb": 18.5,
      "allocated_gb": 5.0,
      "utilization_pct": 20.8
    }
  ]
}
```

#### Supported Languages

```http
GET /languages
```

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

### Language Codes

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `nl` | Dutch |
| `pt` | Portuguese | `cs` | Czech |
| `es` | Spanish | `tr` | Turkish |
| `fr` | French | `ar` | Arabic |
| `de` | German | `zh` | Chinese |
| `it` | Italian | `ja` | Japanese |
| `pl` | Polish | `ko` | Korean |
| `ru` | Russian | `hi` | Hindi |

## Client Examples

### Python Client

#### Basic Usage

```python
import websockets
import json
import wave

async def stream_tts():
    async with websockets.connect("ws://localhost:8002/ws") as ws:
        with wave.open("output.wav", "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            chunks = 0
            total_bytes = 0
            
            text_chunks = [
                "Once upon a time, in a quiet valley surrounded by tall mountains,",
                "there lived a young horse named Spirit.",
                "Spirit was not like the other horses."
            ]
            
            for i, text in enumerate(text_chunks):
                is_last = (i == len(text_chunks) - 1)
                
                message = {
                    "text": text,
                    "language": "en",
                    "end": is_last
                }
                
                print(f"[{i+1}/{len(text_chunks)}] Sending: {text[:50]}...")
                await ws.send(json.dumps(message))
                
                while True:
                    data = await ws.recv()
                    
                    if isinstance(data, str):
                        if data == "DONE":
                            print("Received DONE signal")
                            break
                        elif data.startswith("ERROR"):
                            print(f"Error: {data}")
                            return
                        break
                    
                    elif isinstance(data, bytes):
                        wf.writeframes(data)
                        total_bytes += len(data)
                        chunks += 1
                        print(f"  Received: {len(data):>6d} bytes")
            
            print(f"\nTotal: {chunks} audio chunks, {total_bytes:,} bytes")

import asyncio
asyncio.run(stream_tts())
```

#### LLM Integration Example

When integrating with LLM output streams, **organize text into natural phrases** for best tonality:

```python
import websockets
import json
import asyncio

async def stream_from_llm(llm_response):
    """
    Stream TTS from LLM response.
    
    KEY: Split text into phrases at natural pause points:
    - Periods (.)
    - Commas (,)
    - Question marks (?)
    - Exclamation marks (!)
    - Newlines (\n)
    
    This helps TTS model find proper tonality and pacing.
    """
    async with websockets.connect("ws://localhost:8002/ws") as ws:
        # Split LLM response into natural phrases
        import re
        phrases = re.split(r'([.?!]\s*|\n+)', llm_response)
        
        # Combine phrases with their delimiters
        text_chunks = []
        for i in range(len(phrases)):
            if i > 0 and i < len(phrases) - 1:
                # Combine delimiter with next phrase
                text_chunks.append(phrases[i-1] + phrases[i] + phrases[i+1].strip())
            elif i == len(phrases) - 1 and phrases[i].strip():
                text_chunks.append(phrases[i].strip())
            elif i == 0 and phrases[i].strip():
                text_chunks.append(phrases[i].strip())
        
        # Filter empty chunks
        text_chunks = [c.strip() for c in text_chunks if c.strip()]
        
        # Stream each chunk
        for i, text in enumerate(text_chunks):
            message = {
                "text": text,
                "language": "en",
                "end": i == len(text_chunks) - 1
            }
            
            print(f"[{i+1}/{len(text_chunks)}] {text[:60]}...")
            await ws.send(json.dumps(message))
            
            # Receive and play audio
            while True:
                data = await ws.recv()
                if isinstance(data, bytes):
                    # Process audio (play or save)
                    play_audio(data)
                elif data == "DONE":
                    break
                else:
                    print(f"Message: {data}")
                    break

def play_audio(audio_data):
    """Play audio data in real-time (implement with your audio library)"""
    pass
```

**Best Practices for LLM Integration:**

1. **Phrase at natural pauses**: Split at periods, commas, question marks
2. **Keep chunks 10-50 words**: Too short = choppy, too long = latency
3. **Preserve punctuation**: Helps TTS model with intonation
4. **Avoid splitting mid-sentence**: Breaks tonality
5. **Test with your content**: Different topics may need different chunk sizes

**Example chunking strategies:**

```python
# Strategy 1: Sentence-based (recommended for most content)
import re
chunks = [c.strip() for c in re.split(r'(?<=[.?!])\s+', text) if c.strip()]

# Strategy 2: Clause-based (better for complex sentences)
chunks = [c.strip() for c in re.split(r'(?<=[,;])\s+|(?<=[.?!])\s+', text) if c.strip()]

# Strategy 3: Fixed word count (use with caution)
words = text.split()
chunks = [' '.join(words[i:i+30]) for i in range(0, len(words), 30)]
```

### JavaScript/TypeScript Client

```typescript
interface TTSMessage {
  text: string;
  speaker_wav: string;
  language?: string;
  end?: boolean;
}

async function streamTTS(textChunks: string[], speakerWav: string) {
  const ws = new WebSocket("ws://localhost:8002/ws");
  const audioContext = new AudioContext();
  const audioQueue: AudioBuffer[] = [];
  let isPlaying = false;

  ws.onopen = () => {
    console.log("Connected to TTS server");
  };

  ws.onmessage = async (event: MessageEvent) => {
    if (event.data instanceof Blob) {
      // Audio chunk received
      const arrayBuffer = await event.data.arrayBuffer();
      const audioBuffer = pcmToAudioBuffer(arrayBuffer, audioContext);
      audioQueue.push(audioBuffer);
      playAudioQueue();
    } else {
      // Text message (DONE or ERROR)
      const message = event.data as string;
      if (message === "DONE") {
        console.log("TTS complete");
      } else if (message.startsWith("ERROR")) {
        console.error("TTS Error:", message);
      }
    }
  };

  // Send text chunks
  for (let i = 0; i < textChunks.length; i++) {
    const message: TTSMessage = {
      text: textChunks[i],
      speaker_wav: speakerWav,
      language: "en",
      end: i === textChunks.length - 1
    };
    
    console.log(`Sending chunk ${i + 1}/${textChunks.length}`);
    ws.send(JSON.stringify(message));
    
    // Optional: delay between chunks for more streaming-like behavior
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}

function pcmToAudioBuffer(pcmData: ArrayBuffer, context: AudioContext): AudioBuffer {
  const audioBuffer = context.createAudioBuffer(
    1, // mono
    pcmData.byteLength / 2, // 16-bit = 2 bytes per sample
    24000 // sample rate
  );
  
  const float32 = new Float32Array(pcmData.byteLength / 2);
  const int16 = new Int16Array(pcmData);
  
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768; // Convert to float32 [-1, 1]
  }
  
  audioBuffer.copyToChannel(float32, 0);
  return audioBuffer;
}

async function playAudioQueue() {
  if (isPlaying || audioQueue.length === 0) return;
  
  isPlaying = true;
  
  while (audioQueue.length > 0) {
    const buffer = audioQueue.shift()!;
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start();
    
    await new Promise(resolve => {
      source.onended = resolve;
    });
  }
  
  isPlaying = false;
}

// Usage
const textChunks = [
  "Hello, this is a test.",
  "This is the second chunk.",
  "And this is the final chunk."
];

streamTTS(textChunks, "/path/to/speaker.wav");
```

### cURL Examples

```bash
# Check server health
curl http://localhost:8002/health

# Get GPU statistics
curl http://localhost:8002/gpu-stats

# Get supported languages
curl http://localhost:8002/languages
```

## GPU Management

### Auto-Select GPU

When running with `--auto-select-gpu`, the server checks available memory on all CUDA devices at startup and selects the one with the most free memory:

```
2026-03-12 21:40:20,670 - __main__ - INFO - ============================================================
2026-03-12 21:40:20,670 - __main__ - INFO - GPU Memory Statistics
2026-03-12 21:40:20,670 - __main__ - INFO - ============================================================
2026-03-12 21:40:20,670 - __main__ - INFO - GPU  0: NVIDIA GeForce RTX 3080          | ████████████████████░░░░░░░░░░░░░░░░ |   8.50GB / 10.00GB ( 85.0%)
2026-03-12 21:40:20,670 - __main__ - INFO - GPU  1: NVIDIA GeForce RTX 4090          | ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ |   4.80GB / 24.00GB ( 20.0%) ← SELECTED
2026-03-12 21:40:20,670 - __main__ - INFO - ============================================================
```

### Manual GPU Selection

Force a specific GPU when you know which one is available:

```bash
uv run python main.py --cuda-device 1
```

### Monitoring GPU Usage

Check GPU stats while the server is running:

```bash
curl http://localhost:8002/gpu-stats | jq .
```

## Voice Cloning

### Reference Audio Requirements

- **Format**: WAV, MP3, M4A, or other common audio formats
- **Duration**: 6-30 seconds (optimal: 10-15 seconds)
- **Quality**: Clear speech, minimal background noise
- **Content**: Natural speech (avoid music, sound effects, or extreme emotions)

### Preparing Reference Audio

```bash
# Extract a clean segment from a longer file using ffmpeg
ffmpeg -i source.wav -ss 00:00:10 -t 00:00:15 -ac 1 output.wav

# Normalize audio levels
ffmpeg -i input.wav -af "loudnorm=I=-16" normalized.wav

# Convert to WAV (if needed)
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
```

## Architecture

### Server Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  /ws         │  │  /health     │  │  /gpu-stats      │  │
│  │  (WebSocket) │  │  (HTTP GET)  │  │  (HTTP GET)      │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                   │
│  ┌──────▼───────────────────────────────────────────────┐  │
│  │           ConnectionManager                          │  │
│  │  - Active connection tracking                        │  │
│  │  - TTS model lifecycle                               │  │
│  │  - Audio encoding (24kHz 16-bit PCM)                │  │
│  └─────────────────────────┬───────────────────────────┘  │
│                            │                               │
│  ┌─────────────────────────▼───────────────────────────┐  │
│  │              XTTS-v2 Model                          │  │
│  │  - Loaded on selected GPU (or CPU)                  │  │
│  │  - Multi-language TTS with voice cloning            │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Client                          Server
  │                               │
  │── JSON: {text, speaker_wav}──▶│
  │                               │
  │                               │── Load TTS model (once)
  │                               │── Infer audio
  │                               │── Encode to PCM
  │◀── Binary: audio data ────────│
  │                               │
  │── JSON: {text, end: true} ───▶│
  │                               │
  │◀── Text: "DONE" ──────────────│
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Check GPU memory usage
curl http://localhost:8002/gpu-stats

# Solution: Use a different GPU
uv run python main.py --cuda-device 1

# Or use auto-select
uv run python main.py --auto-select-gpu
```

**2. Model Not Loading**

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Solution: Use CPU fallback
uv run python main.py  # Will use CPU if CUDA fails
```

**3. Audio Quality Issues**

- Ensure reference audio is 6-30 seconds
- Use clean speech samples (no background noise)
- Verify sample rate is 24kHz or will be resampled

**4. Connection Errors**

```bash
# Verify server is running
curl http://localhost:8002/health

# Check firewall/port conflicts
netstat -an | grep 8002
```

### Debug Mode

Enable verbose logging:

```bash
# Set environment variable
export TTS_VERBOSE=1

# Or modify logging in main.py
logging.basicConfig(level=logging.DEBUG)
```

## Performance

### Benchmarks

| Configuration | Inference Time (100 chars) | VRAM Usage | Speed |
|---------------|---------------------------|------------|-------|
| **Default (FP32)** | 1.6-2.0s | ~6-8 GB | 1.0x |
| **FP16 Enabled** | 0.8-1.2s | ~3-5 GB | ~2x faster |
| **FP16 + Speed 1.3** | 0.6-0.9s | ~3-5 GB | ~2.5x faster |
| **FP16 + Speed 1.5** | 0.5-0.7s | ~3-5 GB | ~3x faster |

*Times measured on RTX 2080 Ti, may vary by GPU*

### Performance Optimization Guide

#### 1. Enable FP16 (Recommended)
Half-precision inference provides **2x speedup** with **50% less VRAM usage** and minimal quality impact:

```bash
uv run python main.py --fp16 --auto-select-gpu
```

**Benefits:**
- 2x faster inference
- 50% less VRAM usage (~3-5 GB vs ~6-8 GB)
- Negligible audio quality difference
- **Recommended for all use cases**

#### 2. Adjust Speech Speed
Increase speech speed for faster output without affecting inference time:

```bash
# 30% faster speech
uv run python main.py --speed 1.3

# 50% faster speech
uv run python main.py --speed 1.5
```

**Recommended values:**
- `1.0`: Normal speed (default)
- `1.2-1.3`: Slightly faster (natural sounding)
- `1.4-1.5`: Fast (good for real-time applications)
- `1.6+`: Very fast (may sound robotic)

#### 3. Tune Sampling Parameters
Adjust temperature and sampling for faster/more deterministic output:

```bash
# More deterministic (faster convergence)
uv run python main.py --temperature 0.5 --top-k 25

# Balanced (default)
uv run python main.py --temperature 0.75 --top-k 50

# More varied (slower, more creative)
uv run python main.py --temperature 1.0 --top-k 100
```

**Parameters:**
- `--temperature`: Lower = more deterministic (0.5-1.0)
- `--top-k`: Lower = faster sampling (25-100)
- `--top-p`: Nucleus sampling (0.8-0.95)

#### 4. Optimal Configuration for Real-Time
For best real-time performance (0.2-0.5s per request):

```bash
uv run python main.py --fp16 --speed 1.3 --temperature 0.7 --top-k 40 --auto-select-gpu
```

**Expected performance:**
- 100 characters: ~0.3-0.5s
- 250 characters: ~0.7-1.0s
- 500 characters: ~1.2-1.8s

### Optimization Tips

1. **Enable FP16**: Always use `--fp16` for 2x speedup
2. **Keep server running**: Model stays in memory between requests
3. **Batch small chunks**: Combine short sentences for better efficiency
4. **Use GPU**: CPU inference is ~10x slower
5. **Monitor memory**: Use `/gpu-stats` to prevent OOM errors
6. **Adjust speed**: Use `--speed 1.3` for faster speech output
7. **Tune temperature**: Lower values (0.5-0.7) converge faster

## Security Considerations

**Production Deployment:**

1. **Authentication**: Add WebSocket authentication for production use
2. **Rate Limiting**: Implement request rate limiting per client
3. **Input Validation**: Sanitize text inputs to prevent injection attacks
4. **Network Security**: Bind to specific interfaces, use TLS/SSL
5. **Resource Limits**: Set memory/CPU limits to prevent abuse

Example with authentication:

```python
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    api_key: str = Security(API_KEY_HEADER)
):
    if api_key != os.getenv("TTS_API_KEY"):
        await websocket.close(code=403)
        return
    # ... rest of handler
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for educational and development purposes. Please ensure compliance with Coqui TTS license terms.

## References

- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [XTTS-v2 Model Card](https://huggingface.co/coqui/XTTS-v2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Protocol](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)