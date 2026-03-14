import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from TTS.tts.models.xtts import Xtts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SELECTED_DEVICE: str = "cpu"

# Optimization settings
USE_FP16: bool = False
SPEED_MULTIPLIER: float = 1.0
TEMPERATURE: float = 0.75
LENGTH_PENALTY: float = 1.0
REPETITION_PENALTY: float = 10.0
TOP_K: int = 50
TOP_P: float = 0.85
DO_SAMPLE: bool = True
NUM_BEAMS: int = 1

# Model paths
MODEL_DIR: str = "models/XTTS-v2"


def get_gpu_stats() -> list[dict]:
    """Get memory statistics for all available CUDA devices.

    Returns:
        List of dicts containing GPU stats. Empty list if CUDA unavailable.

    Each dict contains:
        - device: GPU index
        - name: GPU name
        - total_gb: Total VRAM in GB
        - free_gb: Free VRAM in GB
        - allocated_gb: Allocated VRAM in GB
        - utilization_pct: Memory utilization percentage
    """
    if not torch.cuda.is_available():
        return []

    stats = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = props.total_memory / 1e9
        free = total - reserved

        stats.append(
            {
                "device": i,
                "name": props.name,
                "total_gb": round(total, 2),
                "free_gb": round(free, 2),
                "allocated_gb": round(allocated, 2),
                "utilization_pct": round((allocated / total) * 100, 1)
                if total > 0
                else 0.0,
            }
        )
    return stats


def select_best_gpu() -> Optional[int]:
    """Select GPU with most free memory.

    Returns:
        Device index of GPU with most free memory, or None if no GPUs available.
    """
    stats = get_gpu_stats()
    if not stats:
        return None
    best = max(stats, key=lambda x: x["free_gb"])
    logger.info(
        f"Auto-select: GPU {best['device']} has {best['free_gb']}GB free (best)"
    )
    return best["device"]


def log_gpu_stats(selected_device: str) -> None:
    """Log GPU statistics for all devices, marking the selected one.

    Args:
        selected_device: The device string being used (e.g., "cuda:0", "cpu")
    """
    stats = get_gpu_stats()
    if not stats:
        return

    logger.info("=" * 60)
    logger.info("GPU Memory Statistics")
    logger.info("=" * 60)

    selected_idx = None
    if selected_device.startswith("cuda:"):
        try:
            selected_idx = int(selected_device.split(":")[1])
        except (ValueError, IndexError):
            pass

    for s in stats:
        marker = " <- SELECTED" if s["device"] == selected_idx else ""
        bar_width = 40
        util = s["utilization_pct"] / 100
        filled = int(bar_width * util)
        bar = "#" * filled + "-" * (bar_width - filled)

        logger.info(
            f"GPU {s['device']:2d}: {s['name']:<30s} | "
            f"{bar} | "
            f"{s['allocated_gb']:>6.2f}GB / {s['total_gb']:>6.2f}GB "
            f"({s['utilization_pct']:>5.1f}%){marker}"
        )

    logger.info("=" * 60)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace with:
        - cuda_device: Optional CUDA device index (0, 1, etc.)
        - auto_select_gpu: Boolean flag for automatic GPU selection
    """
    parser = argparse.ArgumentParser(
        description="XTTS-v2 Streaming TTS Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Use cuda:0 (default)
  python main.py --cuda-device 1          # Use cuda:1
  python main.py --auto-select-gpu        # Auto-select GPU with most free memory
  python main.py --port 8003              # Change server port
  python main.py --fp16                   # Enable FP16 for faster inference
  python main.py --speed 1.5              # 50%% faster speech
  python main.py --temperature 0.5        # More deterministic output

GPU Selection:
  - If no flag is provided, uses cuda:0
  - --cuda-device N: Force specific GPU (N = 0, 1, 2, ...)
  - --auto-select-gpu: Automatically select GPU with most available memory
  - Flags are mutually exclusive; --auto-select-gpu takes precedence

Performance Optimization:
  - --fp16: Enable half-precision (2x faster, lower VRAM usage)
  - --speed N: Speech speed multiplier (1.0 = normal, 1.5 = 50%% faster)
  - --temperature N: Sampling temp (0.5-1.0, lower = more deterministic)
  - --top-k N: Top-k sampling (default 50)
  - --top-p N: Top-p sampling (default 0.85)
        """,
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Force specific CUDA device index (0, 1, 2, ...). Default: 0",
    )

    parser.add_argument(
        "--auto-select-gpu",
        action="store_true",
        help="Automatically select GPU with most free memory at startup",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Server port number (default: 8002)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )

    # Optimization arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 half-precision for faster inference and lower VRAM usage",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier for speech (1.0 = normal, 1.5 = 50%% faster, etc.)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="Sampling temperature (lower = more deterministic, higher = more varied)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-p (nucleus) sampling parameter",
    )

    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SELECTED_DEVICE, USE_FP16, SPEED_MULTIPLIER, TEMPERATURE, TOP_K, TOP_P

    # Get arguments
    args = get_args()

    # Apply optimization settings from arguments
    USE_FP16 = args.fp16
    SPEED_MULTIPLIER = args.speed
    TEMPERATURE = args.temperature
    TOP_K = args.top_k
    TOP_P = args.top_p

    # Determine device selection
    if torch.cuda.is_available():
        if args.auto_select_gpu:
            device_idx = select_best_gpu()
            if device_idx is not None:
                SELECTED_DEVICE = f"cuda:{device_idx}"
            else:
                SELECTED_DEVICE = "cpu"
                logger.warning("Auto-select failed, falling back to CPU")
        elif args.cuda_device is not None:
            # Validate device index
            if args.cuda_device < torch.cuda.device_count():
                SELECTED_DEVICE = f"cuda:{args.cuda_device}"
            else:
                logger.error(
                    f"CUDA device {args.cuda_device} not available "
                    f"(only {torch.cuda.device_count()} device(s))"
                )
                SELECTED_DEVICE = "cpu"
        else:
            # Default to auto-select GPU with most free memory
            device_idx = select_best_gpu()
            if device_idx is not None:
                SELECTED_DEVICE = f"cuda:{device_idx}"
            else:
                SELECTED_DEVICE = "cpu"
                logger.warning("Auto-select failed, falling back to CPU")
        logger.info(f"Selected device: {SELECTED_DEVICE}")

        # Log all GPU stats
        log_gpu_stats(SELECTED_DEVICE)
    else:
        SELECTED_DEVICE = "cpu"
        logger.info("CUDA not available, using CPU")

    # Log optimization settings
    logger.info("=" * 60)
    logger.info("Optimization Settings")
    logger.info("=" * 60)
    logger.info(f"FP16 enabled: {USE_FP16}")
    logger.info(f"Speed multiplier: {SPEED_MULTIPLIER}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Top-k: {TOP_K}")
    logger.info(f"Top-p: {TOP_P}")
    logger.info("=" * 60)

    logger.info(f"Initializing TTS model on {SELECTED_DEVICE}...")
    await manager.load_tts_model(SELECTED_DEVICE)
    logger.info(f"TTS model ready on {SELECTED_DEVICE}")
    yield
    # Shutdown
    logger.info("Shutting down TTS server...")


app = FastAPI(
    title="XTTS-v2 Streaming TTS Server",
    description="""
## Overview

Real-time Text-to-Speech streaming server using XTTS-v2 model with voice cloning capabilities.

### Features
- **Multi-language support**: 16+ languages (en, pt, es, fr, de, it, etc.)
- **Voice cloning**: Provide a speaker reference WAV file for voice matching
- **Stream-in support**: Send text chunks one at a time, receive audio chunks
- **GPU acceleration**: Automatically uses CUDA if available

### Audio Format
- **Sample rate**: 24kHz
- **Channels**: Mono (1)
- **Bit depth**: 16-bit PCM
- **Output**: Raw binary audio data (no container format)

### WebSocket Protocol

#### Client sends JSON:
```json
{
  "text": "Your text here",
  "speaker_wav": "path/to/speaker.wav",
  "language": "en",
  "end": false
}
```

#### Server responds with:
- Binary audio chunks (24kHz 16-bit PCM)
- Text message "DONE" when `end: true`
- Text message "ERROR: ..." on failure

### Example: Browser Client

```javascript
const ws = new WebSocket('ws://localhost:8002/ws');
const audioContext = new AudioContext();
const audioQueue = [];

ws.onmessage = async (event) => {
  if (event.data instanceof Blob) {
    const audioData = await event.data.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(
      pcmToWav(audioData, 24000)
    );
    playChunk(audioBuffer);
  }
};

// Send text chunks
ws.send(JSON.stringify({
  text: "Hello world",
  speaker_wav: "speaker.wav",
  language: "en",
  end: false
}));
```

### Example: Python Client

```python
import websockets
import json

async with websockets.connect('ws://localhost:8002/ws') as ws:
    # Send chunk
    await ws.send(json.dumps({
        "text": "Hello",
        "speaker_wav": "speaker.wav",
        "language": "en",
        "end": False
    }))
    
    # Receive audio
    data = await ws.recv()
    if isinstance(data, bytes):
        # Process 24kHz 16-bit PCM audio
        pass
```
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.tts_model: Optional[Xtts] = None
        self.speaker_latents: dict[str, tuple] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_audio(self, websocket: WebSocket, data: bytes) -> None:
        await websocket.send_bytes(data)

    def encode_audio_24khz(self, audio: np.ndarray) -> bytes:
        """Encode audio as 24kHz 16-bit PCM"""
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    async def stream_inference(
        self,
        text: str,
        websocket: WebSocket,
        language: str = "en",
    ) -> None:
        """Generate TTS using streaming inference with cached speaker latents"""
        if not self.is_model_loaded():
            logger.error("Model not loaded")
            return

        logger.info(f"Starting streaming inference for: {text[:50]}...")

        try:
            # Get cached speaker latents for language
            if language not in self.speaker_latents:
                logger.warning(f"No speaker latents for language: {language}")
                return

            gpt_cond_latent, speaker_embedding = self.speaker_latents[language]

            if gpt_cond_latent is None or speaker_embedding is None:
                logger.error(f"Speaker latents not loaded for: {language}")
                return

            logger.info(
                f"XTTS streaming: lang={language}, speed={SPEED_MULTIPLIER}, temp={TEMPERATURE}, "
                f"top_k={TOP_K}, top_p={TOP_P}"
            )

            # Run inference in executor to avoid blocking event loop
            def run_inference():
                return self.tts_model.inference_stream(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                )

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, run_inference)

            # Send each chunk immediately as it arrives
            chunk_count = 0
            total_bytes = 0
            start_time = time.time()

            for i, chunk in enumerate(chunks):
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.cpu().numpy()

                encoded = self.encode_audio_24khz(chunk)
                await self.send_audio(websocket, encoded)

                chunk_count += 1
                total_bytes += len(encoded)

                if i == 0:
                    first_chunk_time = time.time() - start_time
                    logger.info(
                        f"First chunk: {first_chunk_time:.3f}s, {len(encoded)} bytes"
                    )

            elapsed = time.time() - start_time
            logger.info(
                f"Sent {chunk_count} chunks, {total_bytes} bytes in {elapsed:.3f}s"
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback

            traceback.print_exc()
            try:
                await websocket.send_text(f"ERROR: {str(e)}")
            except Exception:
                pass

    async def load_tts_model(self, device: str = "cpu") -> None:
        if self.tts_model is not None:
            logger.info("TTS model already loaded")
            return

        logger.info(f"Loading XTTS-v2 model on {device}...")
        logger.info(f"FP16 enabled: {USE_FP16}")
        logger.info(f"Model dir: {MODEL_DIR}")

        # Load config and model from files
        from TTS.tts.configs.xtts_config import XttsConfig

        config = XttsConfig()
        config.load_json(f"{MODEL_DIR}/config.json")

        self.tts_model = Xtts.init_from_config(config)
        self.tts_model.load_checkpoint(
            config,
            checkpoint_dir=MODEL_DIR,
            use_deepspeed=False,
        )

        # Move to device
        self.tts_model.to(device)

        # Apply FP16 quantization for faster inference
        if USE_FP16 and "cuda" in device:
            logger.info("Applying FP16 quantization...")
            self.tts_model.half()

        self.tts_model.eval()

        # Load speaker latents for default speakers
        logger.info("Loading speaker latents...")
        self.speaker_latents["en"] = self._load_speaker_latents("test_speaker_shrt.wav")
        self.speaker_latents["pt"] = self._load_speaker_latents("test_speaker_shrt.wav")

        logger.info("XTTS-v2 model loaded successfully")

    def is_model_loaded(self) -> bool:
        return self.tts_model is not None

    def _load_speaker_latents(self, audio_path: str) -> tuple:
        """Load speaker latents from audio file"""
        import os

        if not os.path.exists(audio_path):
            logger.error(f"Speaker file not found: {audio_path}")
            return None, None

        logger.info(f"Computing speaker latents from: {audio_path}")
        gpt_cond_latent, speaker_embedding = self.tts_model.get_conditioning_latents(
            audio_path=[audio_path]
        )
        logger.info(
            f"Speaker latents loaded: gpt={gpt_cond_latent.shape}, emb={speaker_embedding.shape}"
        )
        return gpt_cond_latent, speaker_embedding

    def unload_model(self) -> None:
        """Unload model and clear GPU memory"""
        if self.tts_model is not None:
            logger.info("Unloading model and clearing GPU memory...")
            self.tts_model = None
            self.tts_api = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("Model unloaded")


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time TTS streaming.

    Protocol:
    1. Client sends JSON messages with text chunks
    2. Server responds with binary audio chunks (24kHz 16-bit PCM)
    3. Set end: true to signal completion (server sends "DONE")

    Message format:
    {
      "text": "Text to convert",
      "speaker_wav": "path/to/reference.wav",
      "language": "en",
      "end": false
    }
    """
    logger.info("WebSocket endpoint called")
    await manager.connect(websocket)
    logger.info("Client connected")

    language = "en"

    try:
        while True:
            logger.info("Waiting for message...")
            data = await websocket.receive_text()
            logger.info(f"Received message: {data[:50]}...")

            if not manager.is_model_loaded():
                logger.error("Model not loaded")
                await websocket.send_text("Error: Model not loaded")
                break

            import json

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.error("Invalid JSON")
                await websocket.send_text("ERROR: Invalid JSON")
                break

            is_end = message.get("end", False)
            text = message.get("text", "")
            language = message.get("language", language)

            if not text:
                if is_end:
                    await websocket.send_text("DONE")
                break

            logger.info(f"Processing chunk: {text[:50]}...")
            await manager.stream_inference(
                text=text,
                websocket=websocket,
                language=language,
            )
            logger.info("Chunk done")

            if is_end:
                await websocket.send_text("DONE")
                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get(
    "/health",
    summary="Health check",
    description="Returns server status and device information",
)
async def health_check():
    """
    Check server health and device info.

    Returns:
    - status: Server status
    - model_loaded: Whether TTS model is loaded
    - active_connections: Number of connected clients
    - device: Current device in use (e.g., "cuda:0", "cpu")
    - cuda_available: Whether CUDA is available
    - cuda_device_count: Number of available CUDA devices
    """
    return {
        "status": "healthy",
        "model_loaded": manager.is_model_loaded(),
        "active_connections": len(manager.active_connections),
        "device": SELECTED_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }


@app.get(
    "/gpu-stats",
    summary="GPU statistics",
    description="Returns detailed memory statistics for all available CUDA devices",
)
async def gpu_stats():
    """
    Get detailed GPU memory statistics.

    Returns:
    - selected_device: Currently selected device
    - cuda_available: Whether CUDA is available
    - device_count: Number of CUDA devices
    - devices: List of GPU stats, each containing:
        - device: GPU index
        - name: GPU name/model
        - total_gb: Total VRAM in GB
        - free_gb: Free VRAM in GB
        - allocated_gb: Currently allocated VRAM in GB
        - utilization_pct: Memory utilization percentage
    """
    stats = get_gpu_stats()
    return {
        "selected_device": SELECTED_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "device_count": len(stats),
        "devices": stats,
    }


@app.get(
    "/languages",
    summary="Supported languages",
    description="Returns list of supported language codes",
)
async def get_languages():
    """
    Get supported language codes.

    Common codes: en, pt, es, fr, de, it, pl, ru, nl, cs, tr, ar, zh, ja, ko, hi
    """
    return {
        "languages": [
            "en",
            "pt",
            "es",
            "fr",
            "de",
            "it",
            "pl",
            "ru",
            "nl",
            "cs",
            "tr",
            "ar",
            "zh",
            "ja",
            "ko",
            "hi",
        ]
    }


if __name__ == "__main__":
    import uvicorn

    args = get_args()
    uvicorn.run(app, host=args.host, port=args.port)
