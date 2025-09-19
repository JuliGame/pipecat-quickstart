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
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional



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
            # "websocket_settings": {
            #     "audio": {
            #         # For Docker on Windows, Attendee container can reach host via host.docker.internal
            #         "url": "",
            #         "sample_rate": sample_rate,
            #     }
            # },
            "voice_agent_settings": {
                "url": "https://constantly-adjusted-pheasant.ngrok-free.app/client/"
            }
        },
    )
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create bot: {status} {data}")
    return data  # expected: { id, meeting_url, state, transcription_state }


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
    async def orchestrate() -> int:
        server = None
        try:
            print("Creating Attendee bot...")
            bot = create_attendee_bot(args.meeting_url, args.bot_name, api_key, sample_rate=16000)
            bot_id = bot.get("id")
            if not bot_id:
                print(f"Unexpected create-bot response: {bot}", file=sys.stderr)
                return 1
            print(f"Created bot id: {bot_id}")
            print("Bot created. voice_bridge.py should receive the WebSocket connection shortly.")
            return 0
        except asyncio.CancelledError:
            return 0
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
    try:
        return asyncio.run(orchestrate())
    except KeyboardInterrupt:
        print("Shutting down.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


