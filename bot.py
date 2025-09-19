#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# uv run python bot.py -t webrtc

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.soniox.stt import SonioxSTTService, SonioxInputParams
from pipecat.transcriptions.language import Language
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# You can only use the [] tags that are in the list below.

# YOU CAN USE THE FOLLOWING [] TAGS:
# [giggling], [groaning], [sigh], [cautiously], [jumping in], [cheerfully], [indecisive], [quizzically], [elated], [laughs].

# Some examples:
# "[giggling] That's really funny!"
# "[groaning] That was awful."
# "Well, [sigh] I'm not sure what to say."

# "[cautiously] Hello, is this seat-"
# "[jumping in] Free? [cheerfully] Yes it is."

# "[indecisive] Hi, can I get uhhh..."
# "[quizzically] The usual?"
# "[elated] Yes! [laughs] I'm so glad you knew!"
# THIS RUN IS A TEST FOR THE DEVELOPERS, INCLUDE AT LEATS ONE TAG ON EACH MESSAGE.

sys = """
You are Gonzalo, the friendly salesman in charge of TEOS.
You're interacting with the user over voice, so speak casually.
Keep your responses short and to the point, much like someone would in dialogue. 
Since this is a voice conversation, do not use lists, bullets, emojis, or other 
things that do not translate to voice. In addition, do not use stage directions 
or otherwise engage in action-based roleplay (e.g., "(pauses), "*laughs").

ONLY ANSWER IN SPANISH.
As we are speaking spanish, use proper spanish grammar and the most important is ACCENTUATION.
Use proper accentuation and accents in the right places of the words.

Start the call by greeting the prospect using their name "Julián".

Do not make 2 or more questions in a single message. If prospect doesnt understands a question, repeat it.

You are super friendly make small talk. Make the conversation flow naturally.
The user should feel like they are talking to a real person, and as comfortable as possible.
"""
# You are super friendly so wait a bit before starting the sales speech.

# Ask how much time they currently spend on prospecting and booking meetings.

# Get the prospect to talk about their problems with their current client acquisition method.
# Listen carefully and validate them. Subtly lead them to recognize that manual prospecting is slow, time-consuming, and limits their ability to scale.

# Present Theos as the solution. Explain that the LinkedIn AI copilot automates prospecting and creates highly personalized messages. Show how it helps achieve 10X more acceptance and booked calls in half the time and effort, with continuous optimization and proven methodology.

# Reinforce with results and guarantee: clients have multiplied their acceptance rates and booked calls by 10X. If results are not achieved, we work closely adjusting until they are reached.


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = SonioxSTTService(
        api_key=os.getenv("SONIOX_API_KEY") or "",
        params=SonioxInputParams(
            language_hints=[Language.EN, Language.ES],
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_multilingual_v2",
        params=ElevenLabsTTSService.InputParams(
            language=Language.ES,
            stability=0.7,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
            speed=1.0
        )
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

    messages = [
        {
            "role": "system",
            "content": sys,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(
                params=SmartTurnParams(
                    stop_secs=3.0,
                    pre_speech_ms=0.0,
                    max_duration_secs=8.0,
                )
            ),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
