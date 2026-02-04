# pip install sounddevice scipy numpy

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
import os

# ---------------------------------
# Load env
# ---------------------------------
load_dotenv()

# ---------------------------------
# Config
# ---------------------------------
SAMPLE_RATE = 16000
DURATION = 5  # seconds
AUDIO_FILE = r"c:\code\agenticai_realpage\whisper\user_input.wav"

# ---------------------------------
# Load models
# ---------------------------------
whisper_model = whisper.load_model("small")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

# ---------------------------------
# Record audio from mic
# ---------------------------------
def record_audio():
    print("Speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("Recording saved")

# ---------------------------------
# Speech → Text
# ---------------------------------
def transcribe_audio(path: str) -> str:
    result = whisper_model.transcribe(
        path,
        language="en",
        fp16=False
    )
    return result["text"]

# ---------------------------------
# GPT Response
# ---------------------------------
def ask_gpt(text: str) -> str:
    response = llm.invoke([
        HumanMessage(content=text)
    ])
    return response.content

# ---------------------------------
# Main Assistant Loop
# ---------------------------------
def run_assistant():
    record_audio()

    print("\nTranscribing...")
    user_text = transcribe_audio(AUDIO_FILE)
    print(f"\nYou said:\n{user_text}")

    print("\nThinking...")
    answer = ask_gpt(user_text)
    print(f"\nAssistant:\n{answer}")

# ---------------------------------
# Run
# ---------------------------------
if __name__ == "__main__":
    run_assistant()
