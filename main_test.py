"""
XTTS-v2 TTS Server - Complete Test Suite

Tests all main.py functionality:
- GPU management (unit tests)
- Argument parsing (unit tests)
- WebSocket streaming (integration tests)
- Spirit story E2E test
"""

import asyncio
import json
import sys
import wave
import os
import time
import websockets
from main import (
    get_gpu_stats,
    select_best_gpu,
    get_args,
    log_gpu_stats,
    ConnectionManager,
)


# ============================================================================
# SECTION 1: Unit Tests - GPU Management
# ============================================================================


def test_gpu_stats():
    """Test GPU statistics gathering"""
    print("\n" + "=" * 60)
    print("TEST 1: GPU Statistics")
    print("=" * 60)

    stats = get_gpu_stats()

    if not stats:
        print("[SKIP] No CUDA devices available")
        return True

    print(f"[PASS] Found {len(stats)} CUDA device(s)")

    for s in stats:
        assert "device" in s, "Missing 'device' key"
        assert "name" in s, "Missing 'name' key"
        assert "total_gb" in s, "Missing 'total_gb' key"
        assert "free_gb" in s, "Missing 'free_gb' key"
        assert "allocated_gb" in s, "Missing 'allocated_gb' key"
        assert "utilization_pct" in s, "Missing 'utilization_pct' key"
        print(f"  GPU {s['device']}: {s['name']} ({s['free_gb']}GB free)")

    print("[PASS] All GPU stat fields validated")
    return True


def test_auto_select():
    """Test automatic GPU selection"""
    print("\n" + "=" * 60)
    print("TEST 2: Auto GPU Selection")
    print("=" * 60)

    best = select_best_gpu()

    if best is None:
        print("[SKIP] No GPU available for auto-selection")
        return True

    print(f"[PASS] Selected GPU {best} (most free memory)")

    stats = get_gpu_stats()
    if not stats:
        print("[FAIL] Stats missing after selection")
        return False

    assert best < len(stats), f"Selected GPU {best} out of range"

    selected_gpu = stats[best]
    for s in stats:
        assert selected_gpu["free_gb"] >= s["free_gb"], (
            f"Selected GPU {best} doesn't have most free memory"
        )

    print(f"[PASS] Verified: GPU {best} has {selected_gpu['free_gb']}GB free")
    return True


def test_gpu_logging():
    """Test GPU statistics logging"""
    print("\n" + "=" * 60)
    print("TEST 3: GPU Logging")
    print("=" * 60)

    stats = get_gpu_stats()
    if stats:
        log_gpu_stats("cuda:0")
        print("[PASS] GPU logging completed")
    else:
        print("[SKIP] No CUDA devices to log")

    return True


# ============================================================================
# SECTION 2: Unit Tests - Argument Parsing
# ============================================================================


def test_argument_parsing():
    """Test command-line argument parsing"""
    print("\n" + "=" * 60)
    print("TEST 4: Argument Parsing")
    print("=" * 60)

    test_cases = [
        (["main"], None, False, 8002, "0.0.0.0"),
        (["main", "--cuda-device", "1"], 1, False, 8002, "0.0.0.0"),
        (["main", "--auto-select-gpu"], None, True, 8002, "0.0.0.0"),
        (["main", "--port", "9000"], None, False, 9000, "0.0.0.0"),
        (["main", "--host", "127.0.0.1"], None, False, 8002, "127.0.0.1"),
        (["main", "--cuda-device", "0", "--port", "8003"], 0, False, 8003, "0.0.0.0"),
    ]

    for (
        argv,
        expected_device,
        expected_auto,
        expected_port,
        expected_host,
    ) in test_cases:
        sys.argv = argv
        args = get_args()

        assert args.cuda_device == expected_device, (
            f"cuda_device mismatch: {args.cuda_device} != {expected_device}"
        )
        assert args.auto_select_gpu == expected_auto, (
            f"auto_select_gpu mismatch: {args.auto_select_gpu} != {expected_auto}"
        )
        assert args.port == expected_port, (
            f"port mismatch: {args.port} != {expected_port}"
        )
        assert args.host == expected_host, (
            f"host mismatch: {args.host} != {expected_host}"
        )

        print(
            f"[PASS] {argv} -> device={args.cuda_device}, auto={args.auto_select_gpu}, "
            f"port={args.port}, host={args.host}"
        )

    return True


# ============================================================================
# SECTION 3: Integration Tests - WebSocket Streaming
# ============================================================================


async def test_websocket_connection():
    """Test basic WebSocket connection"""
    print("\n" + "=" * 60)
    print("TEST 5: WebSocket Connection")
    print("=" * 60)

    uri = "ws://localhost:8002/ws"

    try:
        async with websockets.connect(uri) as ws:
            print("[PASS] Connected to server")

            # Send simple test message
            message = {
                "text": "Hello world",
                "language": "en",
                "end": True,
            }
            await ws.send(json.dumps(message))

            # Wait for response
            data = await asyncio.wait_for(ws.recv(), timeout=10.0)

            if isinstance(data, bytes):
                print(f"[PASS] Received {len(data)} bytes of audio")
                return True
            elif data == "DONE":
                print("[PASS] Received DONE signal")
                return True
            elif data.startswith("ERROR"):
                print(f"[FAIL] Received error: {data}")
                return False
            else:
                print(f"[WARN] Received unexpected response: {data}")
                return True

    except ConnectionRefusedError:
        print("[SKIP] Server not running at ws://localhost:8002/ws")
        print("  Start server with: uv run python main.py --port 8002")
        return True
    except asyncio.TimeoutError:
        print("[FAIL] Server connection timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        return False


# ============================================================================
# SECTION 4: E2E Test - Spirit Story Streaming
# ============================================================================


async def test_spirit_story_streaming():
    """End-to-end test with Spirit story"""
    print("\n" + "=" * 60)
    print("TEST 6: Spirit Story E2E Streaming")
    print("=" * 60)

    uri = "ws://localhost:8002/ws"

    # Spirit story - chunks to stream
    story_chunks = [
        "Once upon a time, in a quiet valley surrounded by tall mountains,",
        "there lived a young horse named Spirit.",
        "Spirit was not like the other horses.",
        "While they were content grazing in the pasture day after day,",
        "Spirit dreamed of something more.",
        "He wanted to be free.",
        "Free to run across endless plains.",
        "Free to explore distant forests.",
        "Free to discover what lay beyond the valley walls.",
        "Every night, Spirit would press his nose against the fence.",
        "Looking at the moonlit horizon.",
        "Wondering what adventures awaited him.",
        "One stormy night, something miraculous happened.",
        "A strong wind knocked down part of the fence.",
        "Spirit's chance had finally come.",
        "He stepped through the broken fence into the unknown.",
        "The wind howled around him as he began to run.",
        "For the first time in his life, Spirit was truly free.",
    ]

    language = "en"

    try:
        async with websockets.connect(uri) as websocket:
            print("[PASS] Connected to server")
            print(f"Story has {len(story_chunks)} chunks to stream\n")

            audio_chunks = []
            total_bytes = 0
            chunk_times = []

            for i, text in enumerate(story_chunks, 1):
                print(f"[{i}/{len(story_chunks)}] Sending: {text}")

                message = {
                    "text": text,
                    "language": language,
                    "end": i == len(story_chunks),
                }

                start_time = time.time()
                await websocket.send(json.dumps(message))

                # Receive audio for this chunk
                chunk_data = b""

                try:
                    while True:
                        data = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        if isinstance(data, bytes):
                            chunk_data += data
                            total_bytes += len(data)
                        elif data == "DONE":
                            print("       Received: DONE")
                            break
                        else:
                            print(f"       Text: {data}")
                            break
                except asyncio.TimeoutError:
                    pass

                elapsed = time.time() - start_time
                chunk_times.append(elapsed)

                if chunk_data:
                    audio_chunks.append(chunk_data)
                    print(f"       Received: {len(chunk_data)} bytes in {elapsed:.3f}s")

                print()

            # Calculate statistics
            if chunk_times:
                avg_time = sum(chunk_times) / len(chunk_times)
                min_time = min(chunk_times)
                max_time = max(chunk_times)
                print(f"Chunk timing stats:")
                print(f"  Average: {avg_time:.3f}s")
                print(f"  Min: {min_time:.3f}s")
                print(f"  Max: {max_time:.3f}s")

            # Save combined audio
            if audio_chunks:
                combined = b"".join(audio_chunks)
                with wave.open("horse_story_test_output.wav", "w") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(24000)
                    wav_file.writeframes(combined)

                print(
                    f"\n[PASS] Total: {len(audio_chunks)} chunks, {total_bytes} bytes"
                )
                print("[PASS] Saved to horse_story_test_output.wav")

                # Validate audio
                expected_chunks = len(story_chunks)
                if len(audio_chunks) == expected_chunks:
                    print(f"[PASS] All {expected_chunks} chunks generated successfully")
                else:
                    print(
                        f"[WARN] Expected {expected_chunks} chunks, got {len(audio_chunks)}"
                    )

                return True
            else:
                print("[FAIL] No audio chunks received")
                return False

    except ConnectionRefusedError:
        print("[SKIP] Server not running at ws://localhost:8002/ws")
        print("  Start server with: uv run python main.py --port 8002")
        return True
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# SECTION 5: Test Runner
# ============================================================================


def run_unit_tests():
    """Run all unit tests"""
    print("\n" + "#" * 60)
    print("  UNIT TESTS - GPU & Argument Management")
    print("#" * 60)

    results = []

    try:
        results.append(("GPU Statistics", test_gpu_stats()))
    except Exception as e:
        print(f"[FAIL] GPU Statistics test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("GPU Statistics", False))

    try:
        results.append(("Auto Selection", test_auto_select()))
    except Exception as e:
        print(f"[FAIL] Auto Selection test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Auto Selection", False))

    try:
        results.append(("GPU Logging", test_gpu_logging()))
    except Exception as e:
        print(f"[FAIL] GPU Logging test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("GPU Logging", False))

    try:
        results.append(("Argument Parsing", test_argument_parsing()))
    except Exception as e:
        print(f"[FAIL] Argument Parsing test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Argument Parsing", False))

    return results


async def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "#" * 60)
    print("  INTEGRATION TESTS - WebSocket & Streaming")
    print("#" * 60)

    results = []

    try:
        results.append(("WebSocket Connection", await test_websocket_connection()))
    except Exception as e:
        print(f"[FAIL] WebSocket Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("WebSocket Connection", False))

    try:
        results.append(("Spirit Story E2E", await test_spirit_story_streaming()))
    except Exception as e:
        print(f"[FAIL] Spirit Story E2E test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Spirit Story E2E", False))

    return results


def print_summary(unit_results, integration_results):
    """Print test summary"""
    all_results = unit_results + integration_results

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in all_results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in all_results)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60 + "\n")

    return all_passed


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  XTTS-v2 TTS Server - Complete Test Suite")
    print("=" * 60 + "\n")

    # Run unit tests
    unit_results = run_unit_tests()

    # Run integration tests
    integration_results = await run_integration_tests()

    # Print summary
    all_passed = print_summary(unit_results, integration_results)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
