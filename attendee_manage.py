"""Manage Attendee bots for meetings.

- Reuse an active bot for a given meeting URL if present
- Otherwise create a new one, pointing at the provided WebSocket URL

Usage:
    uv run attendee_manage.py \
        --meeting-url "https://meet.google.com/abc-def-ghi" \
        --bot-name "Pipecat Bot" \
        --ws-url "ws://host.docker.internal:8765/attendee-websocket" \
        --sample-rate 16000

Outputs a one-line JSON with at least: {"bot_id": "...", "state": "..."}
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

STATE_DB = ".attendee_bots.json"


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
    return read_env("ATTENDEE_API_KEY", "oHvfwfR4knQHINXkTp8qJDFLwWghjh7r") or ""


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


def create_attendee_bot(meeting_url: str, bot_name: str, api_key: str, ws_url: str, sample_rate: int) -> Dict[str, Any]:
    status, data = http_json(
        method="POST",
        path="/api/v1/bots",
        api_key=api_key,
        body={
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "recording_settings": {"format": "none"},
            "websocket_settings": {
                "audio": {
                    "url": ws_url,
                    "sample_rate": sample_rate,
                }
            },
        },
    )
    if status not in (200, 201):
        raise RuntimeError(f"Failed to create bot: {status} {data}")
    return data


def get_bot(bot_id: str, api_key: str) -> Dict[str, Any]:
    status, data = http_json("GET", f"/api/v1/bots/{bot_id}", api_key)
    if status != 200:
        raise RuntimeError(f"Failed to get bot {bot_id}: {status} {data}")
    return data


def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(data: Dict[str, Any]) -> None:
    with open(STATE_DB, "w", encoding="utf-8") as f:
        json.dump(data, f)


def find_or_create_bot(meeting_url: str, bot_name: str, ws_url: str, sample_rate: int, api_key: str) -> Dict[str, Any]:
    state = load_state()
    existing_id = state.get(meeting_url)
    if existing_id:
        try:
            bot = get_bot(existing_id, api_key)
            bot_state = bot.get("state")
            if bot_state and bot_state != "ended":
                return bot
        except Exception:
            pass
        # If ended or fetch failed, remove mapping and create new
        state.pop(meeting_url, None)

    bot = create_attendee_bot(meeting_url, bot_name, api_key, ws_url, sample_rate)
    bot_id = bot.get("id")
    if bot_id:
        state[meeting_url] = bot_id
        save_state(state)
    return bot


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Find or create Attendee bot for a meeting URL")
    parser.add_argument("--meeting-url", required=True, help="Zoom/Meet URL")
    parser.add_argument("--bot-name", default="Pipecat Bot", help="Attendee bot display name")
    parser.add_argument("--ws-url", default=read_env("ATTENDEE_WS_URL", "ws://host.docker.internal:8765/attendee-websocket"), help="Attendee audio WebSocket URL")
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000, 24000], help="Audio sample rate")
    args = parser.parse_args(argv)

    api_key = get_api_key()
    if not api_key:
        print("Missing ATTENDEE_API_KEY", file=sys.stderr)
        return 2

    bot = find_or_create_bot(args.meeting_url, args.bot_name, args.ws_url, args.sample_rate, api_key)
    out = {"bot_id": bot.get("id"), "state": bot.get("state")}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


