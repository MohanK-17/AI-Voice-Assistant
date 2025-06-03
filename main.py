import logging
import os
import datetime
import json
import uuid
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import azure, openai, silero

logger = logging.getLogger("function-calling")
logger.setLevel(logging.INFO)

load_dotenv()

SESSION_ID = str(uuid.uuid4())
SESSION_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
LOG_FILE = "speech_log.json"

# Initialize or update session record
def initialize_session():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Check if session exists already
    session_entry = next((s for s in data if s["session_id"] == SESSION_ID), None)

    if not session_entry:
        session_entry = {
            "session_id": SESSION_ID,
            "session_start_time": SESSION_START_TIME,
            "logs": []
        }
        data.append(session_entry)

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return data


# Log user/assistant message
def log_speech_json(speaker: str, text: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "speaker": speaker.lower(),
        "text": text
    }

    data = initialize_session()

    # Append log to the correct session
    for session in data:
        if session["session_id"] == SESSION_ID:
            session["logs"].append(entry)
            break

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Voice Agent
class FunctionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful assistant communicating through voice. 
                Don't use any unpronounceable characters.
                Note: If asked to print to the console, use the `print_to_console` function.
            """,
            stt=openai.STT.with_azure(
                model="gpt-4o-transcribe",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment="gpt-4o-transcribe",
                api_version="2025-03-01-preview",
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            ),
            llm=openai.LLM.with_azure(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("OPENAI_API_VERSION", "2025-03-01-preview")
            ),
            tts=azure.TTS(
                speech_key=os.getenv("AZURE_OPENAI_API_KEY"),
                speech_region="eastus2",
                voice="ta-IN-PallaviNeural"
            ),
            vad=silero.VAD.load()
        )

    @function_tool
    async def print_to_console(self, context: RunContext):
        print("Console Print Success!")
        return None, "I've printed to the console."

    async def on_enter(self):
        await self.session.generate_reply()

    async def on_response(self, response: str):
        log_speech_json("assistant", response)
        return await super().on_response(response)


# Session entrypoint
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()

    @session.on("user_input_transcribed")
    def on_transcript(transcript):
        if transcript.is_final:
            log_speech_json("user", transcript.transcript)

    initialize_session()

    await session.start(
        agent=FunctionAgent(),
        room=ctx.room
    )


# Run app
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
