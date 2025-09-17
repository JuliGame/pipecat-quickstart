"""Voice bridge with pluggable engines.

Run a local WebSocket server for Attendee realtime audio. Choose the voice engine
at runtime without touching the bot connection. Supports:

- elevenlabs: ElevenLabs SDK streaming to PCM (default)
- pipecat: delegate to the existing bot.py pipeline for TTS (placeholder)

Usage:
    uv run voice_bridge.py --engine elevenlabs --frame-ms 40 --pace

Then separately, create or reuse the Attendee bot with `attendee_manage.py` pointing to the WS URL.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import time
from io import BytesIO
from typing import Any, Optional

import websockets
import urllib.request
import urllib.error
from dotenv import load_dotenv
from elevenlabs import VoiceSettings  # type: ignore
from elevenlabs.client import ElevenLabs  # type: ignore


load_dotenv(override=True)

class ElevenLabsEngine:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY") or "")
        # Default to Adam voice from docs if not provided
        # https://elevenlabs.io/docs/cookbooks/text-to-speech/streaming
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "pNInz6obpgDQGcFmaJgB"

    async def tts_pcm(self, text: str) -> tuple[bytes, int]:
        output_format = f"pcm_{self.sample_rate}"
        print(f"[Bridge:TTS] Using ElevenLabs voice_id={self.voice_id} format={output_format}")
        # SDK path
        try:
            response = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                output_format=output_format,
                text=text,
                model_id="eleven_multilingual_v2",
                voice_settings=VoiceSettings(
                    stability=0.7,
                    similarity_boost=0.8,
                    style=0.5,
                    use_speaker_boost=True,
                    speed=1.0,
                ),
            )
            buf = BytesIO()
            for chunk in response:
                if chunk:
                    buf.write(chunk)
            audio = buf.getvalue()
            if len(audio) % 2 != 0:
                audio = audio[:-1]
            return audio, self.sample_rate
        except Exception as e:
            print(f"[Bridge:TTS] SDK failed ({e}); trying HTTP fallback")

        # HTTP fallback
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
            headers = {
                "xi-api-key": os.getenv("ELEVENLABS_API_KEY") or "",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            }
            body = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.8,
                    "style": 0.5,
                    "use_speaker_boost": True,
                    "speed": 1.0,
                },
                "output_format": output_format,
            }
            req = urllib.request.Request(url=url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                audio = resp.read()
                if len(audio) % 2 != 0:
                    audio = audio[:-1]
                return audio, self.sample_rate
        except Exception as e:
            print(f"[Bridge:TTS] HTTP fallback failed: {e}")
            return b"", self.sample_rate


class VoiceBridgeServer:
    def __init__(self, engine: Any, sample_rate: int, frame_ms: int, pace: bool) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.pace = pace
        self.responded_once = False

    async def handler(self, ws: Any) -> None:
        print("[Bridge] Client connected")
        # Proactively greet on connect to ensure audio output even without incoming audio
        if not self.responded_once:
            try:
                self.responded_once = True
                text = "[cheerfully] Hola Julián, soy Gonzalo de TEOS. ¿Cómo estás?"
                pcm, sr = await self.engine.tts_pcm(text)
                await self._send_pcm(ws, pcm, sr)
            except Exception as e:
                print(f"[Bridge] Initial greet failed: {e}")

        async for message in ws:
            try:
                payload = json.loads(message)
            except Exception:
                continue
            if payload.get("trigger") != "realtime_audio.mixed":
                continue
            if not self.responded_once:
                self.responded_once = True
                text = "[cheerfully] Hola Julián, soy Gonzalo de TEOS. ¿Cómo estás?"
                pcm, sr = await self.engine.tts_pcm(text)
                await self._send_pcm(ws, pcm, sr)

    async def _send_pcm(self, ws: Any, pcm: bytes, sample_rate: int) -> None:
        frame_samples = int(sample_rate * (self.frame_ms / 1000.0))
        frame_bytes = max(2, frame_samples * 2)
        start = time.perf_counter()
        for i in range(0, len(pcm), frame_bytes):
            frame = pcm[i:i+frame_bytes]
            if not frame:
                continue
            chunk_b64 = base64.b64encode(frame).decode("ascii")
            await ws.send(json.dumps({
                "trigger": "realtime_audio.bot_output",
                "data": {"chunk": chunk_b64, "sample_rate": sample_rate},
            }))
            if self.pace:
                target = (i + frame_bytes) / (sample_rate * 2)
                now = time.perf_counter() - start
                delay = target - now
                if delay > 0:
                    await asyncio.sleep(delay)


# Note: For maximum compatibility across websockets versions, we skip process_request
# and let non-WS probes fail the handshake without crashing the server.


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run voice bridge WebSocket server")
    parser.add_argument("--engine", default="elevenlabs", choices=["elevenlabs"], help="Voice engine")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000, 24000])
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("--no-pace", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.engine == "elevenlabs":
        engine = ElevenLabsEngine(args.sample_rate)
    else:
        raise SystemExit("Unsupported engine")

    server = VoiceBridgeServer(engine, args.sample_rate, args.frame_ms, not args.no_pace)
    print(f"[Bridge] Starting on ws://{args.host}:{args.port}/attendee-websocket engine={args.engine} sr={args.sample_rate} frame_ms={args.frame_ms} pace={not args.no_pace}")

    async def orchestrate() -> int:
        ws_server = await websockets.serve(server.handler, args.host, args.port)
        try:
            await asyncio.Future()
        finally:
            ws_server.close()
            await ws_server.wait_closed()
        return 0

    return asyncio.run(orchestrate())


if __name__ == "__main__":
    raise SystemExit(main())


