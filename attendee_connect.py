"""Attendee local connector.

This script connects to a locally running Attendee instance (http://localhost:8000)
to create a meeting bot and then exits. No polling or transcript fetching is performed.

Run this side-by-side with your Pipecat voice agent (`uv run bot.py`) to handle
live audio streaming and transcription via the Pipecat reference server config
already present in `bot.py`.

Usage examples:

    uv run attendee_connect.py --meeting-url "https://us05web.zoom.us/j/123..." --bot-name "Pipecat Bot"

Environment variables (optional):
    ATTENDEE_BASE_URL   Defaults to http://localhost:8000
    ATTENDEE_API_KEY    API key for your local Attendee instance
    MEETING_URL         Default meeting URL if --meeting-url is not provided

Notes:
- This script does not poll Attendee for status or transcripts.
- Live transcription is expected to be produced by your Pipecat pipeline in `bot.py`.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import math
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import websockets
from http import HTTPStatus

from bot import sys as BOT_SYSTEM_PROMPT  # type: ignore
from io import BytesIO
from elevenlabs import VoiceSettings  # type: ignore
from elevenlabs.client import ElevenLabs  # type: ignore


def read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    return default


def get_base_url() -> str:
    base = read_env("ATTENDEE_BASE_URL", "http://localhost:8000")
    return base if base is not None else "http://localhost:8000"


def get_api_key() -> str:
    # Fallback to user-provided key if env not set
    key = read_env("ATTENDEE_API_KEY", "oHvfwfR4knQHINXkTp8qJDFLwWghjh7r") or ""
    print(f"[ATTENDEE] API key: {key}")
    return key


def build_request(method: str, url: str, api_key: str, data: Optional[Dict[str, Any]] = None) -> urllib.request.Request:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}",
    }
    payload = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url=url, data=payload, headers=headers, method=method)
    return req


def http_json(method: str, path: str, api_key: str, body: Optional[Dict[str, Any]] = None):
    base = get_base_url().rstrip("/")
    url = f"{base}{path}"
    req = build_request(method, url, api_key, data=body)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            raw = resp.read().decode("utf-8")
            try:
                return status, json.loads(raw)
            except json.JSONDecodeError:
                return status, raw
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        raise RuntimeError(f"HTTP {e.code} for {path}: {err_body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error for {path}: {e}")


def create_attendee_bot(meeting_url: str, bot_name: str, api_key: str, sample_rate: int) -> Dict[str, Any]:
    status, data = http_json(
        method="POST",
        path="/api/v1/bots",
        api_key=api_key,
        body={
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "recording_settings": {"format": "none"},
            # Configure realtime audio websocket per Attendee docs
            # https://docs.attendee.dev/guides/realtime-audio-input-and-output
            "websocket_settings": {
                "audio": {
                    # For Docker on Windows, Attendee container can reach host via host.docker.internal
                    "url": "ws://host.docker.internal:8765/attendee-websocket",
                    "sample_rate": sample_rate,
                }
            },
        },
    )
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create bot: {status} {data}")
    return data  # expected: { id, meeting_url, state, transcription_state }


def _http_post_json(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    req = urllib.request.Request(url=url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def openai_chat_complete(messages: list[dict[str, str]], model: str = "gpt-4.1-nano", max_tokens: int = 64, temperature: float = 0.6) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    return "Hola Julián, ¿cómo estás hoy? Aguanten los sanguches de milanesa, soy gabriel"
    if not api_key:
        return "Hola Julián, ¿cómo estás hoy?"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        data = _http_post_json(url, headers, body)
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Hola Julián, ¿cómo estás hoy?"


_ELEVEN_CLIENT: Any = None


def _get_eleven_client() -> Any:
    global _ELEVEN_CLIENT
    if _ELEVEN_CLIENT is None:
        api_key = os.getenv("ELEVENLABS_API_KEY") or ""
        if api_key:
            _ELEVEN_CLIENT = ElevenLabs(api_key=api_key)
    return _ELEVEN_CLIENT


def elevenlabs_tts_pcm16(text: str, sample_rate: int = 16000) -> tuple[bytes, int]:
    api_key = os.getenv("ELEVENLABS_API_KEY") or ""
    voice_id = os.getenv("ELEVENLABS_VOICE_ID") or ""
    if not api_key or not voice_id:
        return b""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/octet-stream",
    }
    # ElevenLabs supports pcm_16000/pcm_24000
    output_format = f"pcm_{sample_rate}"
    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.8,
            "style": 0.5,
            "use_speaker_boost": True,
        },
        "output_format": output_format,
    }
    req = urllib.request.Request(url=url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    # Prefer official SDK to ensure correct format
    try:
        client = _get_eleven_client()
        if client:
            voice_id = os.getenv("ELEVENLABS_VOICE_ID") or ""
            if voice_id:
                output_format = f"pcm_{sample_rate}"
                response = client.text_to_speech.convert(
                    voice_id=voice_id,
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
                print(f"[TTS] SDK bytes {len(audio)} @ {sample_rate} Hz ({output_format})")
                return audio, sample_rate
    except Exception as e:
        print(f"[TTS] SDK path failed: {e}. Falling back to HTTP")

    # Fallback to raw HTTP
    try:
        with urllib.request.urlopen(req) as resp:
            audio = resp.read()
            if audio.startswith(b"RIFF") and len(audio) > 44:
                try:
                    wav_pcm, wav_sr = extract_wav_pcm16_le(audio)
                    print(f"[TTS] Extracted WAV PCM {len(wav_pcm)} bytes @ {wav_sr} Hz from ElevenLabs")
                    audio = wav_pcm
                    sample_rate = wav_sr
                except Exception as e:
                    print(f"[TTS] WAV parse failed, using raw bytes: {e}")
            if len(audio) % 2 != 0:
                audio = audio[:-1]
            print(f"[TTS] HTTP bytes {len(audio)} @ {sample_rate} Hz")
            return audio, sample_rate
    except Exception as e:
        print(f"[TTS] ElevenLabs HTTP request failed; returning silence: {e}")
        return b"", sample_rate


class SimpleVoiceAgent:
    async def generate_reply_text(self, user_hint: str) -> str:
        messages = [
            {"role": "system", "content": BOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_hint},
        ]
        text = openai_chat_complete(messages)
        print(f"[LLM] Reply text: {text}")
        return text

    async def synthesize_pcm16(self, text: str, sample_rate: int = 16000) -> tuple[bytes, int]:
        # Force 16 kHz end-to-end to avoid resampling artifacts
        chosen_sr = 16000
        if sample_rate != chosen_sr:
            print(f"[TTS] Forcing TTS sample rate to {chosen_sr} (incoming {sample_rate})")
        pcm, actual_sr = elevenlabs_tts_pcm16(text, chosen_sr)
        print(f"[TTS] PCM length: {len(pcm)} bytes at {actual_sr} Hz")
        return pcm, actual_sr


class AttendeeAudioWebSocket:
    def __init__(
        self,
        voice_agent: SimpleVoiceAgent,
        expected_sample_rate: int = 16000,
        send_test_tone_ms: int = 0,
        byteswap_output: bool = False,
        downmix_stereo: bool = False,
        attenuate: float = 1.0,
        frame_ms: int = 40,
        pace_realtime: bool = True,
    ) -> None:
        self.voice_agent = voice_agent
        self.expected_sample_rate = expected_sample_rate
        self.responded_once = False
        self.send_test_tone_ms = send_test_tone_ms
        self.byteswap_output = byteswap_output
        self.downmix_stereo = downmix_stereo
        self.attenuate = max(0.0, min(attenuate, 1.0))
        self.frame_ms = max(5, min(frame_ms, 200))
        self.pace_realtime = pace_realtime

    async def handler(self, ws: Any) -> None:
        try:
            path = getattr(ws, "path", "")
        except Exception:
            path = ""
        print(f"[WS] Client connected path={path}")
        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print("[WS] Non-JSON message received; ignoring")
                continue

            trigger = payload.get("trigger")
            # print(f"[WS] Trigger: {trigger}")
            if trigger != "realtime_audio.mixed":
                continue

            data = payload.get("data", {})
            chunk_b64 = data.get("chunk")
            sample_rate = data.get("sample_rate") or self.expected_sample_rate
            # print(f"[WS] Incoming audio sample_rate={sample_rate}, chunk_b64_len={len(chunk_b64) if chunk_b64 else 0}")
            try:
                _ = base64.b64decode(chunk_b64) if chunk_b64 else b""
                # if _:
                    # print(f"[WS] Decoded PCM len={len(_)} bytes")
            except Exception:
                print("[WS] Failed to base64-decode incoming chunk")
                _ = b""

            # Minimal loop: reply once to confirm audio bridge works
            if not self.responded_once:
                self.responded_once = True
                if self.send_test_tone_ms > 0:
                    print(f"[TEST] Sending {self.send_test_tone_ms} ms 440Hz tone @ {self.expected_sample_rate} Hz")
                    pcm = generate_sine_pcm16(self.expected_sample_rate, 440.0, self.send_test_tone_ms, amplitude=0.3)
                    out_sr = self.expected_sample_rate
                else:
                    reply_text = await self.voice_agent.generate_reply_text(
                        "[cheerfully] Saluda brevemente a Julián y preséntate."
                    )
                    pcm, out_sr = await self.voice_agent.synthesize_pcm16(reply_text, sample_rate)
                if pcm:
                    await self._send_pcm(ws, pcm, out_sr)
                else:
                    print("[WS] No PCM generated; skipping send")

    async def _send_pcm(self, ws: Any, pcm: bytes, sample_rate: int) -> None:
        # Optional transforms for diagnostics
        original_len = len(pcm)
        if self.downmix_stereo and (len(pcm) % 4 == 0):
            pcm = downmix_stereo_to_mono_le(pcm)
            print(f"[AUDIO] Downmixed stereo->mono: {original_len} -> {len(pcm)} bytes")
            original_len = len(pcm)
        if self.byteswap_output:
            pcm = byteswap16(pcm)
            print(f"[AUDIO] Byteswapped 16-bit samples: len={len(pcm)}")
        if self.attenuate < 1.0 and len(pcm) >= 2:
            pcm = attenuate_pcm16_le(pcm, self.attenuate)
            print(f"[AUDIO] Attenuated by {self.attenuate}: len={len(pcm)}")

        # Send in configurable frames (default 40 ms). 16 kHz mono 16-bit => 640 bytes per 20 ms
        frame_samples = int(sample_rate * (self.frame_ms / 1000.0))
        frame_bytes = frame_samples * 2
        total = len(pcm)
        if frame_bytes <= 0:
            # Fallback to 20 ms
            frame_bytes = (int(sample_rate * 0.02)) * 2
        num_frames = (total + frame_bytes - 1) // frame_bytes
        sent = 0
        print(f"[WS] Sending bot_output in {num_frames} frames (frame_ms={self.frame_ms}, frame_bytes={frame_bytes}, total={total}, pace={self.pace_realtime})")
        start_time = time.perf_counter()
        for i in range(0, total, frame_bytes):
            frame = pcm[i : i + frame_bytes]
            if not frame:
                continue
            chunk_b64 = base64.b64encode(frame).decode("ascii")
            msg = {
                "trigger": "realtime_audio.bot_output",
                "data": {
                    "chunk": chunk_b64,
                    "sample_rate": sample_rate,
                },
            }
            await ws.send(json.dumps(msg))
            sent += len(frame)
            # Pace the stream to ~real-time
            if self.pace_realtime:
                # Target time from start for next frame
                target_s = (i + frame_bytes) / (sample_rate * 2)  # bytes -> seconds
                now = time.perf_counter() - start_time
                delay = target_s - now
                if delay > 0:
                    await asyncio.sleep(delay)
        print(f"[WS] Completed send: {sent} bytes over {num_frames} frames at {sample_rate} Hz")


async def run_audio_server(host: str = "0.0.0.0", port: int = 8765):
    voice_agent = SimpleVoiceAgent()
    # Runtime flags are injected when constructing this in main(); this default is used when called directly
    attendee_ws = AttendeeAudioWebSocket(voice_agent)
    print(f"[WS] Starting server on ws://{host}:{port}/attendee-websocket")
    return await websockets.serve(attendee_ws.handler, host, port, ping_interval=20, ping_timeout=20, process_request=_http_health_or_path)


async def _http_health_or_path(path: str, request_headers: Any):
    # If it's not a websocket upgrade, respond 200 OK for health checks
    upgrade = (request_headers.get("Upgrade") or "").lower()
    if upgrade != "websocket":
        body = b"OK\n"
        return HTTPStatus.OK, [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))], body
    # Enforce path
    if path != "/attendee-websocket":
        body = b"Not Found\n"
        return HTTPStatus.NOT_FOUND, [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))], body


def generate_sine_pcm16(sample_rate: int, freq_hz: float, duration_ms: int, amplitude: float = 0.5) -> bytes:
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    max_amp = int(32767 * max(0.0, min(amplitude, 1.0)))
    out = bytearray()
    for n in range(num_samples):
        t = n / sample_rate
        s = int(max_amp * math.sin(2 * math.pi * freq_hz * t))
        # little-endian 16-bit signed
        out.append(s & 0xFF)
        out.append((s >> 8) & 0xFF)
    return bytes(out)


def extract_wav_pcm16_le(wav_bytes: bytes) -> tuple[bytes, int]:
    # Minimal WAV parser for PCM16 LE
    # RIFF header: 12 bytes, then chunks; find 'fmt ' then 'data'
    if not wav_bytes.startswith(b"RIFF") or b"WAVE" not in wav_bytes[8:12]:
        raise ValueError("Not a WAV file")
    pos = 12
    fmt_sample_rate = None
    num_channels = None
    bits_per_sample = None
    data_start = None
    data_size = None
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos+4]
        chunk_size = int.from_bytes(wav_bytes[pos+4:pos+8], "little", signed=False)
        pos += 8
        if chunk_id == b"fmt ":
            if pos + chunk_size > len(wav_bytes):
                break
            audio_format = int.from_bytes(wav_bytes[pos:pos+2], "little")
            num_channels = int.from_bytes(wav_bytes[pos+2:pos+4], "little")
            fmt_sample_rate = int.from_bytes(wav_bytes[pos+4:pos+8], "little")
            bits_per_sample = int.from_bytes(wav_bytes[pos+14:pos+16], "little")
            if audio_format not in (1,):
                raise ValueError(f"Unsupported WAV format code {audio_format}")
        elif chunk_id == b"data":
            data_start = pos
            data_size = chunk_size
            break
        pos += chunk_size
    if data_start is None or data_size is None:
        raise ValueError("No data chunk in WAV")
    if fmt_sample_rate is None or num_channels is None or bits_per_sample is None:
        raise ValueError("Invalid WAV fmt chunk")
    if bits_per_sample != 16:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")
    raw = wav_bytes[data_start:data_start+data_size]
    # If stereo, downmix to mono
    if num_channels == 2:
        raw = downmix_stereo_to_mono_le(raw)
    elif num_channels != 1:
        raise ValueError(f"Unsupported channels: {num_channels}")
    return raw, fmt_sample_rate


def byteswap16(pcm: bytes) -> bytes:
    if len(pcm) < 2:
        return pcm
    out = bytearray(len(pcm))
    out[0::2] = pcm[1::2]
    out[1::2] = pcm[0::2]
    return bytes(out)


def downmix_stereo_to_mono_le(pcm: bytes) -> bytes:
    # Interpret as interleaved little-endian 16-bit stereo [L0,R0,L1,R1,...]
    out = bytearray()
    for i in range(0, len(pcm), 4):
        l = pcm[i] | (pcm[i + 1] << 8)
        r = pcm[i + 2] | (pcm[i + 3] << 8)
        # Convert to signed 16-bit
        if l >= 32768:
            l -= 65536
        if r >= 32768:
            r -= 65536
        m = (l + r) // 2
        if m < 0:
            m += 65536
        out.append(m & 0xFF)
        out.append((m >> 8) & 0xFF)
    return bytes(out)


def attenuate_pcm16_le(pcm: bytes, factor: float) -> bytes:
    out = bytearray(len(pcm))
    for i in range(0, len(pcm), 2):
        s = pcm[i] | (pcm[i + 1] << 8)
        if s >= 32768:
            s -= 65536
        s = int(s * factor)
        if s < 0:
            s += 65536
        out[i] = s & 0xFF
        out[i + 1] = (s >> 8) & 0xFF
    return bytes(out)

def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Create Attendee bot and bridge realtime audio via WebSocket")
    parser.add_argument("--meeting-url", default=read_env("MEETING_URL"), help="Zoom/Meet URL")
    parser.add_argument("--bot-name", default="Pipecat Bot", help="Attendee bot display name")
    parser.add_argument("--test-tone-ms", type=int, default=0, help="Send a 440Hz test tone for N ms instead of TTS")
    parser.add_argument("--byteswap", action="store_true", help="Byteswap 16-bit samples before sending")
    parser.add_argument("--downmix-stereo", action="store_true", help="Downmix stereo 16-bit PCM to mono before sending")
    parser.add_argument("--attenuate", type=float, default=1.0, help="Scale PCM amplitude (0.0-1.0)")
    parser.add_argument("--create-only", action="store_true", help="Only create the Attendee bot; do not start a local WS server (use when voice_bridge.py is running)")
    args = parser.parse_args(argv)

    if not args.meeting_url:
        print("--meeting-url is required (or set MEETING_URL)", file=sys.stderr)
        return 2

    base_url = get_base_url()
    api_key = get_api_key()
    if not api_key:
        print("Missing ATTENDEE_API_KEY (and no fallback provided)", file=sys.stderr)
        return 2

    print(f"Using Attendee at: {base_url}")
    if not args.create_only:
        print("Starting local WebSocket server for realtime audio on ws://0.0.0.0:8765/attendee-websocket ...")

    async def orchestrate() -> int:
        server = None
        try:
            if not args.create_only:
                # Build server with runtime flags
                voice_agent = SimpleVoiceAgent()
                attendee_ws = AttendeeAudioWebSocket(
                    voice_agent,
                    expected_sample_rate=16000,
                    send_test_tone_ms=args.test_tone_ms,
                    byteswap_output=args.byteswap,
                    downmix_stereo=args.downmix_stereo,
                    attenuate=args.attenuate,
                )
                print(f"[CFG] test_tone_ms={args.test_tone_ms} byteswap={args.byteswap} downmix_stereo={args.downmix_stereo} attenuate={args.attenuate}")
                server = await websockets.serve(attendee_ws.handler, "0.0.0.0", 8765, ping_interval=20, ping_timeout=20)

            print("Creating Attendee bot...")
            bot = create_attendee_bot(args.meeting_url, args.bot_name, api_key, sample_rate=16000)
            bot_id = bot.get("id")
            if not bot_id:
                print(f"Unexpected create-bot response: {bot}", file=sys.stderr)
                return 1
            print(f"Created bot id: {bot_id}")
            if args.create_only:
                print("Bot created. voice_bridge.py should receive the WebSocket connection shortly.")
                return 0
            print("Waiting for Attendee to stream audio to our WebSocket...")
            await asyncio.Future()
        except asyncio.CancelledError:
            return 0
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
        return 0

    try:
        return asyncio.run(orchestrate())
    except KeyboardInterrupt:
        print("Shutting down.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


