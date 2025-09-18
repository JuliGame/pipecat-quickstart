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
from typing import Any, Optional, cast
import threading
import sys

import websockets
import urllib.request
import urllib.error
from dotenv import load_dotenv
from elevenlabs import VoiceSettings  # type: ignore
from elevenlabs.client import ElevenLabs  # type: ignore
# Avoid importing the full bot or heavy Pipecat stacks here to prevent side effects

# Pipecat imports (used when --engine=pipecat)
try:
    from pipecat.frames.frames import (
        InputAudioRawFrame,
        OutputAudioRawFrame,
        TTSAudioRawFrame,
        SpeechOutputAudioRawFrame,
        InterimTranscriptionFrame,
        StartFrame,
        LLMRunFrame,
        LLMTextFrame,
        TTSSpeakFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.soniox.stt import SonioxSTTService, SonioxInputParams
    from pipecat.transcriptions.language import Language
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.transports.base_transport import BaseTransport
    _HAS_PIPECAT = True
except Exception:
    _HAS_PIPECAT = False

# Optional local audio monitoring for debugging
try:
    import simpleaudio as sa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sa = None  # type: ignore

# Optional device introspection
try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sd = None  # type: ignore


load_dotenv(override=True)

# Ensure a compatible event loop policy on Windows for websockets
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

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


def _pcm16le_to_wav_bytes(pcm: bytes, sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    data_size = len(pcm)
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    riff_size = 36 + data_size
    header = bytearray()
    header += b"RIFF"
    header += riff_size.to_bytes(4, "little")
    header += b"WAVE"
    header += b"fmt "
    header += (16).to_bytes(4, "little")  # fmt chunk size
    header += (1).to_bytes(2, "little")   # PCM format
    header += num_channels.to_bytes(2, "little")
    header += sample_rate.to_bytes(4, "little")
    header += byte_rate.to_bytes(4, "little")
    header += block_align.to_bytes(2, "little")
    header += bits_per_sample.to_bytes(2, "little")
    header += b"data"
    header += data_size.to_bytes(4, "little")
    return bytes(header) + pcm


def _http_multipart_openai_transcribe(wav_bytes: bytes, api_key: str, model: str = "whisper-1") -> str:
    boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
    crlf = "\r\n"
    parts: list[bytes] = []
    # model field
    parts.append((
        f"--{boundary}{crlf}Content-Disposition: form-data; name=\"model\"{crlf}{crlf}{model}{crlf}".encode("utf-8")
    ))
    # file field
    parts.append((
        f"--{boundary}{crlf}Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"{crlf}Content-Type: audio/wav{crlf}{crlf}".encode("utf-8")
    ))
    parts.append(wav_bytes)
    parts.append(crlf.encode("utf-8"))
    # end
    parts.append((f"--{boundary}--{crlf}".encode("utf-8")))
    body = b"".join(parts)
    req = urllib.request.Request(
        url="https://api.openai.com/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode("utf-8")
        try:
            data = json.loads(raw)
            # OpenAI returns { text: "..." }
            return (data.get("text") or "").strip()
        except Exception:
            return ""


def _openai_chat_complete(messages: list[dict[str, str]], model: str = "gpt-4.1", max_tokens: int = 90, temperature: float = 0.6) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        return "[cheerfully] Hola, ¿cómo estás?"
    req = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "[cheerfully] Hola, ¿cómo estás?"


class PipecatStackEngine:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._tts = ElevenLabsEngine(sample_rate)

    async def stt_llm_tts(self, pcm16le: bytes, in_sample_rate: int) -> tuple[bytes, int]:
        try:
            # 1) STT via OpenAI Whisper on WAV
            wav = _pcm16le_to_wav_bytes(pcm16le, in_sample_rate)
            openai_key = os.getenv("OPENAI_API_KEY") or ""
            text = _http_multipart_openai_transcribe(wav, openai_key)
            if not text:
                print("[Pipecat] STT returned empty text")
                return b"", self.sample_rate
            print(f"[Pipecat] STT: {text}")
            # 2) LLM with Spanish system prompt (aligned with bot persona)
            system_prompt = (
                "Eres Gonzalo, un vendedor amable de TEOS. SOLO RESPONDES EN ESPAÑOL con acentuación correcta. "
                "Habla de forma breve y conversacional (sin listas, ni emojis, ni indicaciones escénicas). "
                "Saluda usando el nombre del prospecto 'Julián' al inicio de la conversación. "
                "Haz una sola pregunta por turno y mantén un tono cercano."
            )
            reply = _openai_chat_complete([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ])
            print(f"[Pipecat] LLM: {reply}")
            # 3) TTS via ElevenLabs
            pcm, sr = await self._tts.tts_pcm(reply)
            return pcm, sr
        except Exception as e:
            print(f"[Pipecat] Pipeline error: {e}")
            return b"", self.sample_rate


# Exact-pipeline transport and engine using Pipecat (parity with bot.py)
if _HAS_PIPECAT:
    class _AttendeeInputProcessor(FrameProcessor):
        def __init__(self, *, name: Optional[str] = None) -> None:
            super().__init__(name=name)
            self._started_event: asyncio.Event = asyncio.Event()

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            if isinstance(frame, StartFrame):
                try:
                    self._started_event.set()
                    print("[PipecatDebug] InputProcessor: StartFrame received; started_event set")
                except Exception:
                    pass
            # Transparently forward all frames downstream
            try:
                # print(f"[PipecatDebug] InputProcessor forwarding frame: {getattr(frame, 'name', type(frame).__name__)}")
                await self.push_frame(frame, direction)
            except Exception:
                pass

        async def wait_started(self) -> None:
            await self._started_event.wait()

    class _AttendeeOutputProcessor(FrameProcessor):
        def __init__(self, sender_cb: Any, started_event: asyncio.Event, expected_sr: int, *, name: Optional[str] = None) -> None:
            super().__init__(name=name)
            self._sender_cb = sender_cb
            self._started_event = started_event
            self._expected_sr = int(expected_sr)

        def _resample_linear(self, pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
            if src_sr == dst_sr or len(pcm) < 2:
                return pcm
            # 16-bit mono little-endian
            import math
            num_src = len(pcm) // 2
            # Convert to ints
            src = memoryview(pcm)
            def read_sample(idx: int) -> int:
                i = idx * 2
                s = src[i] | (src[i+1] << 8)
                if s >= 32768:
                    s -= 65536
                return s
            ratio = float(src_sr) / float(dst_sr)
            num_dst = max(1, int(num_src / ratio))
            out = bytearray(num_dst * 2)
            for n in range(num_dst):
                pos = n * ratio
                i0 = int(math.floor(pos))
                i1 = min(i0 + 1, num_src - 1)
                frac = pos - i0
                s0 = read_sample(i0)
                s1 = read_sample(i1)
                s = int((1.0 - frac) * s0 + frac * s1)
                if s < 0:
                    s += 65536
                j = n * 2
                out[j] = s & 0xFF
                out[j+1] = (s >> 8) & 0xFF
            return bytes(out)

        async def process_frame(self, frame: Any, direction: Any) -> None:
            await super().process_frame(frame, direction)
            try:
                if isinstance(frame, StartFrame):
                    try:
                        self._started_event.set()
                        print("[PipecatDebug] OutputProcessor: StartFrame received; pipeline_started set")
                    except Exception:
                        pass
                if isinstance(frame, (OutputAudioRawFrame, TTSAudioRawFrame, SpeechOutputAudioRawFrame)):
                    # Send PCM bytes back via provided callback
                    if callable(self._sender_cb):
                        audio = frame.audio
                        sr = int(getattr(frame, "sample_rate", self._expected_sr) or self._expected_sr)
                        try:
                            size = len(audio) if isinstance(audio, (bytes, bytearray, memoryview)) else 0
                            print(f"[PipecatDebug] OutputProcessor: audio frame type={type(frame).__name__} bytes={size} sr={sr}")
                        except Exception:
                            pass
                        if sr != self._expected_sr:
                            try:
                                audio = self._resample_linear(audio, sr, self._expected_sr)
                                sr = self._expected_sr
                            except Exception as e:
                                print(f"[PipecatDebug] OutputProcessor: resample error: {e}")
                        # Ensure even bytes
                        if len(audio) % 2 != 0:
                            audio = audio[:-1]
                        try:
                            await self._sender_cb(audio, sr)
                        except Exception as e:
                            print(f"[PipecatDebug] OutputProcessor: sender_cb error: {e}")
            except Exception as e:
                print(f"[PipecatTransport] Output send error: {e}")
            # Transparently forward all frames downstream
            try:
                # print(f"[PipecatDebug] OutputProcessor forwarding frame: {getattr(frame, 'name', type(frame).__name__)}")
                await self.push_frame(frame, direction)
            except Exception:
                pass

    class AttendeeBridgeTransport(BaseTransport):
        def __init__(self, expected_sr: int) -> None:
            super().__init__(name="AttendeeBridgeTransport")
            self._sender_cb: Any = None
            self._input_proc = _AttendeeInputProcessor(name="attendee-input")
            self._pipeline_started: asyncio.Event = asyncio.Event()
            self._output_proc = _AttendeeOutputProcessor(self._send_out, self._pipeline_started, expected_sr, name="attendee-output")
            self._buffer: list[tuple[bytes, int, int]] = []

        def input(self) -> FrameProcessor:
            return self._input_proc

        def output(self) -> FrameProcessor:
            return self._output_proc

        def set_sender(self, sender_cb: Any) -> None:
            self._sender_cb = sender_cb
            # print("[PipecatDebug] Transport: sender callback set")

        async def _send_out(self, audio: bytes, sample_rate: int) -> None:
            if callable(self._sender_cb):
                # print(f"[PipecatDebug] Transport._send_out: len={len(audio)} sr={sample_rate}")
                await self._sender_cb(audio, sample_rate)

        async def feed_pcm(self, pcm: bytes, sample_rate: int, num_channels: int = 1) -> None:
            if not self._pipeline_started.is_set():
                # Buffer until StartFrame has reached output
                self._buffer.append((pcm, sample_rate, num_channels))
                return
            frame = InputAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=num_channels)
            await self._input_proc.push_frame(frame)

        async def wait_started(self) -> None:
            await self._input_proc.wait_started()
            print("[PipecatDebug] Transport: input started")

        async def wait_pipeline_started(self) -> None:
            await self._pipeline_started.wait()
            print("[PipecatDebug] Transport: pipeline started")

        async def flush_buffer(self) -> None:
            if not self._buffer:
                return
            queued = self._buffer
            self._buffer = []
            for pcm, sr, ch in queued:
                frame = InputAudioRawFrame(audio=pcm, sample_rate=sr, num_channels=ch)
                await self._input_proc.push_frame(frame)

    class PipecatPipelineEngine:
        def __init__(self, sample_rate: int) -> None:
            self.sample_rate = sample_rate
            self.transport = AttendeeBridgeTransport(expected_sr=sample_rate)
            self._runner: Any = None
            self._task: Any = None
            # Persona/context matching bot.py
            try:
                from bot import sys as BOT_SYSTEM_PROMPT  # type: ignore
                self._system_prompt = BOT_SYSTEM_PROMPT
            except Exception:
                self._system_prompt = (
                    "You are Gonzalo, the friendly salesman in charge of TEOS. "
                    "ONLY ANSWER IN SPANISH with correct accentuation. Keep responses short, no lists/emojis." 
                    "Greet the prospect using the name 'Julián' to start."
                )

        async def start(self) -> None:
            # Services mirror bot.py
            stt = SonioxSTTService(
                api_key=os.getenv("SONIOX_API_KEY") or "",
                params=SonioxInputParams(language_hints=[Language.EN, Language.ES]),
            )
            llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")
            _el_api_key = cast(str, os.getenv("ELEVENLABS_API_KEY") or "")
            _el_voice_id = cast(str, os.getenv("ELEVENLABS_VOICE_ID") or "pNInz6obpgDQGcFmaJgB")
            tts = ElevenLabsTTSService(
                api_key=_el_api_key,
                voice_id=_el_voice_id,
                model="eleven_flash_v2_5",
                params=ElevenLabsTTSService.InputParams(
                    language=Language.ES,
                    stability=0.7,
                    similarity_boost=0.8,
                    style=0.5,
                    use_speaker_boost=True,
                    speed=1.0,
                ),
            )

            messages: Any = [
                {"role": "system", "content": self._system_prompt},
            ]
            context = OpenAILLMContext(messages)
            context_agg = llm.create_context_aggregator(context)

            rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

            class _SmartTurnEmulator(FrameProcessor):
                def __init__(self, stop_secs: float = 0.8, *, name: Optional[str] = None) -> None:
                    super().__init__(name=name)
                    self._stop_secs = float(max(0.2, stop_secs))
                    self._last_activity: float | None = None
                    self._watch_task: Any | None = None
                    self._speaking = False

                async def process_frame(self, frame: Any, direction: Any) -> None:
                    await super().process_frame(frame, direction)
                    try:
                        # Forward original frame downstream
                        await self.push_frame(frame, direction)
                    except Exception:
                        pass
                    try:
                        # Start watchdog when pipeline starts
                        if isinstance(frame, StartFrame) and self._watch_task is None:
                            self._watch_task = self.create_task(self._watchdog(), name="smartturn-watchdog")
                            print(f"[PipecatDebug] SmartTurn: watchdog started stop_secs={self._stop_secs}")
                        # Track interim transcription activity
                        if isinstance(frame, InterimTranscriptionFrame):
                            text = str(getattr(frame, "text", "") or "").strip()
                            is_final = bool(getattr(frame, "is_final", False) or getattr(frame, "final", False))
                            now = time.monotonic()
                            if text:
                                self._last_activity = now
                                self._speaking = True
                            if is_final and text:
                                # If the STT marks final explicitly, trigger immediately
                                await self.push_frame(LLMRunFrame())
                                self._speaking = False
                                self._last_activity = None
                    except Exception:
                        pass

                async def _watchdog(self) -> None:
                    try:
                        while True:
                            await asyncio.sleep(0.05)
                            if self._speaking and self._last_activity is not None:
                                elapsed = time.monotonic() - self._last_activity
                                if elapsed >= self._stop_secs:
                                    try:
                                        await self.push_frame(LLMRunFrame())
                                    except Exception:
                                        pass
                                    self._speaking = False
                                    self._last_activity = None
                    except asyncio.CancelledError:
                        return

            smart_turn = _SmartTurnEmulator(stop_secs=0.8, name="smart-turn")

            class _LLMToTTSSpeak(FrameProcessor):
                def __init__(self, *, name: Optional[str] = None) -> None:
                    super().__init__(name=name)
                    self._buffer: str = ""
                    self._debounce: Any | None = None

                async def _finalize_and_speak(self) -> None:
                    try:
                        text = self._buffer.strip()
                        self._buffer = ""
                        if text:
                            print(f"[PipecatDebug] LLM→TTS: finalized len={len(text)}")
                            await self.push_frame(TTSSpeakFrame(text))
                    except Exception:
                        pass
                async def process_frame(self, frame: Any, direction: Any) -> None:
                    await super().process_frame(frame, direction)
                    # Forward original frame
                    try:
                        await self.push_frame(frame, direction)
                    except Exception:
                        pass
                    # Bridge plain LLM text to TTS speak
                    try:
                        fname = type(frame).__name__
                        # Accumulate streaming deltas, then speak once after a brief pause
                        if isinstance(frame, LLMTextFrame):
                            delta = str(getattr(frame, "text", "") or "")
                            if delta:
                                print(f"[PipecatDebug] LLM→TTS: delta len={len(delta)}")
                                self._buffer += delta
                                # debounce finalize
                                try:
                                    if self._debounce is not None:
                                        self._debounce.cancel()
                                except Exception:
                                    pass
                                try:
                                    async def _debounced() -> None:
                                        try:
                                            await asyncio.sleep(0.25)
                                            await self._finalize_and_speak()
                                        except asyncio.CancelledError:
                                            return
                                    self._debounce = self.create_task(_debounced(), name="llm-tts-debounce")
                                except Exception:
                                    pass
                            return

                        # Do not trigger TTS on non-final tokens beyond the accumulator
                        # Some LLMs may emit message frames instead of LLMTextFrame
                        # Try to extract assistant text from LLMMessages* frames
                        if fname in ("LLMMessagesFrame", "LLMMessagesAppendFrame", "LLMMessagesUpdateFrame"):
                            messages = getattr(frame, "messages", None)
                            if isinstance(messages, (list, tuple)) and messages:
                                # Find last assistant content
                                for m in reversed(messages):
                                    role = (m.get("role") if isinstance(m, dict) else None) or ""
                                    content = m.get("content") if isinstance(m, dict) else None
                                    if role == "assistant" and isinstance(content, str) and content.strip():
                                        print(f"[PipecatDebug] LLM→TTS: assistant message len={len(content.strip())}")
                                        # Cancel pending debounce and flush buffer
                                        try:
                                            if self._debounce is not None:
                                                self._debounce.cancel()
                                        except Exception:
                                            pass
                                        if self._buffer.strip():
                                            self._buffer += " " + content.strip()
                                            await self._finalize_and_speak()
                                        else:
                                            await self.push_frame(TTSSpeakFrame(content.strip()))
                                        break
                        # Fallback: ignore non-final frames to avoid spamming TTS
                    except Exception:
                        pass

            llm_to_tts = _LLMToTTSSpeak(name="llm-to-tts")

            pipeline = Pipeline(
                [
                    self.transport.input(),
                    rtvi,
                    stt,
                    smart_turn,
                    context_agg.user(),
                    llm,
                    llm_to_tts,
                    tts,
                    self.transport.output(),
                    context_agg.assistant(),
                ]
            )

            self._task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    enable_metrics=True,
                    enable_usage_metrics=True,
                ),
                observers=[RTVIObserver(rtvi)],
            )
            self._runner = PipelineRunner(handle_sigint=False)
            # Run in background
            asyncio.create_task(self._runner.run(self._task))
            # Wait for StartFrame to propagate so we can feed audio safely
            try:
                await self.transport.wait_started()
            except Exception:
                pass
            # Also wait for StartFrame to reach the output end
            try:
                await self.transport.wait_pipeline_started()
            except Exception:
                pass
            try:
                await self.transport.flush_buffer()
            except Exception:
                pass
            print("[PipecatDebug] Engine: pipeline started and ready")

        async def stop(self) -> None:
            try:
                if self._task is not None:
                    await self._task.cancel()
            except Exception:
                pass

        async def on_client_connected(self) -> None:
            # Kick off greeting
            try:
                if self._task is not None:
                    await self._task.queue_frames([LLMRunFrame()])
            except Exception as e:
                print(f"[Pipecat] on_client_connected error: {e}")

        async def on_audio_chunk(self, pcm: bytes, sample_rate: int) -> None:
            try:
                await self.transport.feed_pcm(pcm, sample_rate)
            except Exception as e:
                print(f"[Pipecat] feed error: {e}")

def _monitor_device_and_beep(sample_rate: int) -> None:
    if not sa:
        print("[Bridge:Monitor] simpleaudio not available; skipping test tone")
        return
    try:
        if 'sd' in globals() and sd is not None:
            default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            devices = sd.query_devices()
            device_info = devices[default_out] if isinstance(default_out, int) and 0 <= default_out < len(devices) else sd.query_devices(None, 'output')
            print(f"[Bridge:Monitor] Output device: {device_info.get('name')} (index={default_out})")
        else:
            print("[Bridge:Monitor] sounddevice not available; using system default output (simpleaudio)")
        # Prefer a safe Windows-native beep in debug/Windows to avoid driver issues
        if os.name == "nt":
            try:
                import winsound  # type: ignore
                winsound.Beep(440, 300)
                print("[Bridge:Monitor] Test tone (winsound) done")
                return
            except Exception:
                pass
        # 300ms 440Hz test tone at moderate volume via simpleaudio elsewhere
        duration_s = 0.3
        sr = sample_rate
        num_samples = int(sr * duration_s)
        samples = bytearray()
        amplitude = 8000
        for n in range(num_samples):
            t = n / sr
            val = int(amplitude * math.sin(2 * math.pi * 440.0 * t))
            samples += int(val).to_bytes(2, byteorder='little', signed=True)
        play_obj = sa.play_buffer(bytes(samples), 1, 2, sr)
        try:
            play_obj.wait_done()
        except Exception:
            pass
        print("[Bridge:Monitor] Test tone (simpleaudio) done")
    except Exception as e:
        print(f"[Bridge:Monitor] device/test tone error: {e}")


class VoiceBridgeServer:
    def __init__(self, engine: Any, sample_rate: int, frame_ms: int, pace: bool, monitor_input: bool, greet_enabled: bool, bot_beep_on_connect: bool = False, bot_tone_hz: int = 0, log_latency: bool = False, debug_trigger: bool = False) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.pace = pace
        self.monitor_input = monitor_input
        self.greet_enabled = greet_enabled
        self.bot_beep_on_connect = bot_beep_on_connect
        self.bot_tone_hz = max(0, int(bot_tone_hz))
        self.log_latency = log_latency
        self.debug_trigger = debug_trigger
        self.responded_once = False
        self._in_pcm = bytearray()
        self._chunks_received = 0
        self._bytes_received = 0
        self._sd_stream = None  # type: ignore[var-annotated]
        self._sd_stream_rate: int | None = None
        self._tone_task: Any | None = None
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0
        if self.monitor_input and sa is None:
            print("[Bridge:Monitor] simpleaudio not available; --monitor-input disabled")
            self.monitor_input = False

    def _ensure_sd_stream(self, samplerate: int) -> bool:
        if sd is None:
            return False
        try:
            if self._sd_stream is not None and self._sd_stream_rate == samplerate:
                return True
            # Recreate if rate changed
            if self._sd_stream is not None:
                try:
                    self._sd_stream.stop()
                    self._sd_stream.close()
                except Exception:
                    pass
                self._sd_stream = None
                self._sd_stream_rate = None
            self._sd_stream = sd.RawOutputStream(
                samplerate=samplerate,
                channels=1,
                dtype='int16',
                blocksize=0,
            )
            stream = self._sd_stream
            if stream is not None:
                stream.start()
                self._sd_stream_rate = samplerate
            print(f"[Bridge:Monitor] sounddevice stream opened @ {samplerate} Hz")
            return True
        except Exception as e:
            print(f"[Bridge:Monitor] sounddevice open failed: {e}")
            try:
                stream3 = self._sd_stream
                if stream3 is not None:
                    stream3.stop()
                    stream3.close()
            except Exception:
                pass
            self._sd_stream = None
            self._sd_stream_rate = None
            return False

    async def handler(self, ws: Any) -> None:
        print("[Bridge] Client connected")
        # Optional: start debug key trigger loop to bypass VAD/Smart Turn
        debug_stop_event: Any = None
        debug_thread: Any = None
        if self.debug_trigger:
            try:
                loop = asyncio.get_running_loop()
                debug_stop_event = threading.Event()
                def _debug_key_loop() -> None:
                    try:
                        try:
                            import msvcrt  # type: ignore
                            _has_msvcrt = True
                        except Exception:
                            _has_msvcrt = False
                        print("[Bridge:Debug] Press 't' for direct TTS test, 'l' to trigger LLM turn, 'q' to stop debug loop")
                        while not debug_stop_event.is_set():
                            ch: str | None = None
                            if _has_msvcrt:
                                if msvcrt.kbhit():  # type: ignore
                                    b = msvcrt.getch()  # type: ignore
                                    try:
                                        ch = b.decode('utf-8', errors='ignore').lower()
                                    except Exception:
                                        ch = None
                                else:
                                    time.sleep(0.05)
                                    continue
                            else:
                                # Fallback (blocking); avoid spinning if not Windows
                                ch = sys.stdin.read(1).lower()
                            if not ch:
                                continue
                            if ch == 'q':
                                print("[Bridge:Debug] Stopping debug key loop")
                                break
                            if ch == 't':
                                print("[Bridge:Debug] Direct TTS test trigger (always using ElevenLabsEngine)")
                                def _do_tts() -> Any:
                                    async def _coro() -> None:
                                        try:
                                            text = "[debug] Hola, esto es una prueba de TTS."
                                            try:
                                                direct = ElevenLabsEngine(self.sample_rate)
                                                pcm, sr = await direct.tts_pcm(text)
                                                await self._send_pcm(ws, pcm, sr)
                                            except Exception as e2:
                                                print(f"[Bridge:Debug] Direct ElevenLabs error: {e2}")
                                        except Exception as e:
                                            print(f"[Bridge:Debug] TTS trigger error: {e}")
                                    return asyncio.run_coroutine_threadsafe(_coro(), loop)
                                _do_tts()
                            if ch == 'T':
                                print("[Bridge:Debug] Bypass pipeline: ElevenLabsEngine direct TTS")
                                def _do_direct() -> Any:
                                    async def _coro2() -> None:
                                        try:
                                            text = "[debug-direct] Esto es TTS directo por ElevenLabs."
                                            try:
                                                direct = ElevenLabsEngine(self.sample_rate)
                                                pcm, sr = await direct.tts_pcm(text)
                                                await self._send_pcm(ws, pcm, sr)
                                            except Exception as e2:
                                                print(f"[Bridge:Debug] Direct ElevenLabs error: {e2}")
                                        except Exception as e:
                                            print(f"[Bridge:Debug] Direct trigger error: {e}")
                                    return asyncio.run_coroutine_threadsafe(_coro2(), loop)
                                _do_direct()
                            if ch == 'l':
                                print("[Bridge:Debug] LLM turn trigger (LLMRunFrame)")
                                if _HAS_PIPECAT and isinstance(self.engine, PipecatPipelineEngine) and self.engine._task is not None:
                                    try:
                                        asyncio.run_coroutine_threadsafe(self.engine._task.queue_frames([LLMRunFrame()]), loop)
                                    except Exception as e:
                                        print(f"[Bridge:Debug] LLM trigger error: {e}")
                                else:
                                    print("[Bridge:Debug] LLM trigger requires PipecatPipelineEngine")
                    except Exception as e:
                        print(f"[Bridge:Debug] Key loop error: {e}")
                debug_thread = threading.Thread(target=_debug_key_loop, name="bridge-debug-keys", daemon=True)
                debug_thread.start()
            except Exception as e:
                print(f"[Bridge:Debug] Failed to start key loop: {e}")
        # If monitoring, announce output device and play a short test tone
        if self.monitor_input:
            _monitor_device_and_beep(self.sample_rate)
        # Optionally send a short tone into the meeting to verify outbound audio
        if self.bot_beep_on_connect:
            try:
                # 300ms 880Hz tone
                duration_s = 0.3
                sr = self.sample_rate
                num_samples = int(sr * duration_s)
                samples = bytearray()
                amplitude = 8000
                for n in range(num_samples):
                    t = n / sr
                    val = int(amplitude * math.sin(2 * math.pi * 880.0 * t))
                    samples += int(val).to_bytes(2, byteorder='little', signed=True)
                await self._send_pcm(ws, bytes(samples), self.sample_rate)
                print("[Bridge] Sent bot_output test beep (300ms @ 880Hz)")
            except Exception as e:
                print(f"[Bridge] Failed to send bot_output test beep: {e}")
        # Start continuous tone loop if configured
        if self.bot_tone_hz > 0 and self._tone_task is None:
            self._tone_task = asyncio.create_task(self._tone_loop(ws))
        # If using the exact Pipecat pipeline, hook sender and start runner
        if _HAS_PIPECAT and isinstance(self.engine, PipecatPipelineEngine):
            try:
                async def sender_cb(pcm: bytes, sr: int) -> None:
                    await self._send_pcm(ws, pcm, sr)
                self.engine.transport.set_sender(sender_cb)
                await self.engine.start()
                if self.greet_enabled:
                    await self.engine.on_client_connected()
            except Exception as e:
                print(f"[Bridge] Pipecat start error: {e}")
        else:
            # Proactively greet on connect to ensure audio output even without incoming audio
            if self.greet_enabled and not self.responded_once and hasattr(self.engine, "tts_pcm"):
                try:
                    text = "[cheerfully] Hola Julián, soy Gonzalo de TEOS. ¿Cómo estás?"
                    pcm, sr = await self.engine.tts_pcm(text)  # type: ignore[attr-defined]
                    await self._send_pcm(ws, pcm, sr)
                    self.responded_once = True
                except Exception as e:
                    print(f"[Bridge] Initial greet failed: {e}")
            elif self.greet_enabled and not hasattr(self.engine, "tts_pcm"):
                print("[Bridge] Greeting skipped: current engine has no tts_pcm")

        try:
            async for message in ws:
                try:
                    payload = json.loads(message)
                except Exception:
                    continue
                if payload.get("trigger") != "realtime_audio.mixed":
                    continue
                # Accumulate incoming audio
                data = payload.get("data", {})
                chunk_b64 = data.get("chunk")
                in_sr = data.get("sample_rate") or self.sample_rate
                ts_ms = data.get("timestamp_ms")
                try:
                    chunk = base64.b64decode(chunk_b64) if chunk_b64 else b""
                except Exception:
                    chunk = b""
                if chunk:
                    if self.log_latency and isinstance(ts_ms, (int, float)):
                        now_ms = int(time.time() * 1000)
                        delta_ms = max(0, now_ms - int(ts_ms))
                        self._latency_sum_ms += float(delta_ms)
                        self._latency_count += 1
                    self._chunks_received += 1
                    self._bytes_received += len(chunk)
                    if self._chunks_received == 1 or (self._chunks_received % 50 == 0):
                        seconds = self._bytes_received / (2 * in_sr)
                        if self.log_latency and self._latency_count > 0:
                            avg_ms = self._latency_sum_ms / float(self._latency_count)
                            print(f"[Bridge] Receiving audio: chunks={self._chunks_received} total={self._bytes_received}B ~{seconds:.2f}s @ {in_sr} Hz | arrival_latency_avg={avg_ms:.1f} ms")
                        else:
                            print(f"[Bridge] Receiving audio: chunks={self._chunks_received} total={self._bytes_received}B ~{seconds:.2f}s @ {in_sr} Hz")
                    self._in_pcm.extend(chunk)
                    # If using Pipecat pipeline, feed audio continuously and skip ad-hoc trigger
                    if _HAS_PIPECAT and isinstance(self.engine, PipecatPipelineEngine):
                        try:
                            await self.engine.on_audio_chunk(chunk, in_sr)
                        except Exception as e:
                            print(f"[Bridge] Pipecat feed error: {e}")
                        continue
                    # Optional: play back what the bot is hearing (local monitoring)
                    if self.monitor_input:
                        # Prefer sounddevice stream if available
                        if sd is not None and self._ensure_sd_stream(in_sr):
                            try:
                                # Ensure even number of bytes
                                if len(chunk) % 2 != 0:
                                    chunk = chunk[:-1]
                                stream = self._sd_stream
                                if stream is not None:
                                    stream.write(chunk)
                            except Exception as e:
                                print(f"[Bridge:Monitor] sd write failed: {e}")
                                try:
                                    stream2 = self._sd_stream
                                    if stream2 is not None:
                                        stream2.stop()
                                        stream2.close()
                                except Exception:
                                    pass
                                self._sd_stream = None
                                self._sd_stream_rate = None
                        elif sa is not None:
                            try:
                                if len(chunk) % 2 != 0:
                                    chunk = chunk[:-1]
                                sa.play_buffer(chunk, 1, 2, in_sr)
                            except Exception as e:
                                # Do not disrupt the bridge on playback errors
                                print(f"[Bridge:Monitor] playback error: {e}")
                # Simple trigger: when > 0.8s accumulated and we haven't responded yet
                if not self.responded_once and len(self._in_pcm) >= int(self.sample_rate * 2 * 0.8):
                    self.responded_once = True
                    input_pcm = bytes(self._in_pcm)
                    self._in_pcm.clear()
                    try:
                        if hasattr(self.engine, "stt_llm_tts"):
                            out_pcm, out_sr = await self.engine.stt_llm_tts(input_pcm, self.sample_rate)
                            if out_pcm:
                                await self._send_pcm(ws, out_pcm, out_sr)
                        else:
                            text = "[cheerfully] Hola Julián, soy Gonzalo de TEOS. ¿Cómo estás?"
                            pcm, sr = await self.engine.tts_pcm(text)
                            await self._send_pcm(ws, pcm, sr)
                    except Exception as e:
                        print(f"[Bridge] STT/LLM/TTS failed: {e}")
        except Exception as e:
            print(f"[Bridge] Client connection error/closed: {e}")
        finally:
            # Stop debug key loop
            try:
                if debug_stop_event is not None:
                    debug_stop_event.set()
            except Exception:
                pass
            # Stop tone loop
            try:
                if self._tone_task is not None:
                    self._tone_task.cancel()
                    self._tone_task = None
            except Exception:
                pass
            # Close sounddevice stream if open
            try:
                stream4 = self._sd_stream
                if stream4 is not None:
                    stream4.stop()
                    stream4.close()
            except Exception:
                pass
            self._sd_stream = None
            self._sd_stream_rate = None
            # Stop Pipecat pipeline if running
            try:
                if _HAS_PIPECAT and isinstance(self.engine, PipecatPipelineEngine):
                    await self.engine.stop()
            except Exception:
                pass

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

    async def _tone_loop(self, ws: Any) -> None:
        freq = float(self.bot_tone_hz)
        sr = int(self.sample_rate)
        frame_samples = max(1, int(sr * (self.frame_ms / 1000.0)))
        amplitude = 9000
        phase = 0.0
        phase_inc = 2.0 * math.pi * freq / sr
        print(f"[Bridge] Continuous tone enabled: {freq:.1f} Hz @ {sr} Hz frame_ms={self.frame_ms}")
        start = time.perf_counter()
        sent_frames = 0
        try:
            while True:
                buf = bytearray(frame_samples * 2)
                idx = 0
                for _ in range(frame_samples):
                    val = int(amplitude * math.sin(phase))
                    if val < 0:
                        val += 1 << 16
                    buf[idx] = val & 0xFF
                    buf[idx + 1] = (val >> 8) & 0xFF
                    idx += 2
                    phase += phase_inc
                    if phase > 2.0 * math.pi:
                        phase -= 2.0 * math.pi
                chunk_b64 = base64.b64encode(bytes(buf)).decode("ascii")
                await ws.send(json.dumps({
                    "trigger": "realtime_audio.bot_output",
                    "data": {"chunk": chunk_b64, "sample_rate": sr},
                }))
                sent_frames += 1
                if self.pace:
                    target = sent_frames * (frame_samples / sr)
                    now = time.perf_counter() - start
                    delay = target - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(self.frame_ms / 1000.0)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[Bridge] Tone loop error: {e}")


# Note: For maximum compatibility across websockets versions, we skip process_request
# and let non-WS probes fail the handshake without crashing the server.


def main(argv: Optional[list[str]] = None) -> int:
    try:
        parser = argparse.ArgumentParser(description="Run voice bridge WebSocket server")
        parser.add_argument("--engine", default="elevenlabs", choices=["elevenlabs", "pipecat"], help="Voice engine")
        parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000, 24000])
        parser.add_argument("--frame-ms", type=int, default=40)
        parser.add_argument("--no-pace", action="store_true")
        parser.add_argument("--monitor-input", action="store_true", help="Play incoming attendee audio locally for debugging")
        parser.add_argument("--no-greet", action="store_true", help="Disable proactive greeting/output; useful for monitoring-only")
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--loop-mode", choices=["auto", "thread", "asyncio"], default="auto", help="Event loop strategy")
        parser.add_argument("--bot-beep-on-connect", action="store_true", help="Send a short test beep into the meeting on client connect")
        parser.add_argument("--bot-tone-hz", type=int, default=0, help="Continuously emit a tone into the meeting at this frequency (0=off)")
        parser.add_argument("--log-latency", action="store_true", help="Log arrival latency vs Attendee timestamp_ms to diagnose delay sources")
        parser.add_argument("--debug-trigger", action="store_true", help="Enable console key triggers: 't' for TTS, 'l' for LLM")
        args = parser.parse_args(argv)

        engine: Any
        if args.engine == "elevenlabs":
            engine = ElevenLabsEngine(args.sample_rate)
        elif args.engine == "pipecat":
            engine = PipecatPipelineEngine(args.sample_rate) if _HAS_PIPECAT else PipecatStackEngine(args.sample_rate)
        else:
            raise SystemExit("Unsupported engine")

        server = VoiceBridgeServer(
            engine,
            args.sample_rate,
            args.frame_ms,
            not args.no_pace,
            args.monitor_input,
            not args.no_greet,
            bot_beep_on_connect=args.bot_beep_on_connect,
            bot_tone_hz=args.bot_tone_hz,
            log_latency=args.log_latency,
            debug_trigger=args.debug_trigger,
        )
        # Decide loop mode
        loop_mode = args.loop_mode
        if loop_mode == "auto":
            loop_mode = "asyncio" if ("debugpy" in sys.modules) else "thread"
        print(f"[Bridge] Starting on ws://{args.host}:{args.port}/attendee-websocket engine={args.engine} sr={args.sample_rate} frame_ms={args.frame_ms} pace={not args.no_pace} greet={not args.no_greet} monitor={args.monitor_input} loop_mode={loop_mode} bot_beep_on_connect={args.bot_beep_on_connect} bot_tone_hz={args.bot_tone_hz} log_latency={args.log_latency}")
        if args.monitor_input:
            print("[Bridge:Monitor] Startup device check + test tone...")
            # _monitor_device_and_beep(args.sample_rate)

        if loop_mode == "asyncio":
            async def orchestrate() -> int:
                try:
                    print(f"[Bridge] Binding WebSocket server on {args.host}:{args.port} ...")
                    async with websockets.serve(
                        server.handler,
                        args.host,
                        args.port,
                        ping_interval=None,
                        ping_timeout=None,
                    ):
                        print(f"[Bridge] WebSocket server is listening at ws://{args.host}:{args.port}/attendee-websocket (Ctrl+C to stop)")
                        await asyncio.Event().wait()
                except Exception as e:
                    print(f"[Bridge] Server error: {e}")
                return 0
            return asyncio.run(orchestrate())
        else:
            # thread mode
            try:
                loop = asyncio.new_event_loop()
            except Exception as e:
                print(f"[Bridge] Failed to create event loop ({e}); falling back to asyncio mode")
                async def orchestrate2() -> int:
                    try:
                        print(f"[Bridge] Binding WebSocket server on {args.host}:{args.port} ...")
                        async with websockets.serve(
                            server.handler,
                            args.host,
                            args.port,
                            ping_interval=None,
                            ping_timeout=None,
                        ):
                            print(f"[Bridge] WebSocket server is listening at ws://{args.host}:{args.port}/attendee-websocket (Ctrl+C to stop)")
                            await asyncio.Event().wait()
                    except Exception as ee:
                        print(f"[Bridge] Server error: {ee}")
                    return 0
                return asyncio.run(orchestrate2())

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_run_loop, name="voice-bridge-loop", daemon=True)
            t.start()

            async def _start_server() -> Any:
                return await websockets.serve(
                    server.handler,
                    args.host,
                    args.port,
                    ping_interval=None,
                    ping_timeout=None,
                )

            ws_server: Any | None = None
            try:
                print(f"[Bridge] Binding WebSocket server on {args.host}:{args.port} ...")
                fut = asyncio.run_coroutine_threadsafe(_start_server(), loop)
                ws_server = fut.result(timeout=10)
                print(f"[Bridge] WebSocket server is listening at ws://{args.host}:{args.port}/attendee-websocket (Ctrl+C to stop)")
                # Keep main thread alive
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("[Bridge] KeyboardInterrupt: shutting down...")
            except Exception as e:
                print(f"[Bridge] Server error: {e}")
            finally:
                try:
                    if ws_server is not None:
                        ws_server.close()
                        asyncio.run_coroutine_threadsafe(ws_server.wait_closed(), loop).result(timeout=5)
                except Exception:
                    pass
                try:
                    loop.call_soon_threadsafe(loop.stop)
                    t.join(timeout=5)
                except Exception:
                    pass
                print("[Bridge] Server loop ended")
            return 0
    except Exception as e:
        print(f"[Bridge] Server error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


