import logging
import os
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import azure, openai, silero

logger = logging.getLogger("function-calling")
logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv()

class FunctionAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful assistant communicating through voice. Don't use any unpronounceable characters.
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

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=FunctionAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
