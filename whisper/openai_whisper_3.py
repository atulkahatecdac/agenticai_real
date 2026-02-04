# pip install pyttsx3

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import pyttsx3
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
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
AUDIO_FILE = "user_input.wav"

# ---------------------------------
# Load models
# ---------------------------------
whisper_model = whisper.load_model("small")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)

# ---------------------------------
# Conversation memory
# ---------------------------------
conversation = []

# ---------------------------------
# Record audio
# ---------------------------------
def record_audio():
    print("\nSpeak now...")
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
    return result["text"].strip()

# ---------------------------------
# GPT with memory
# ---------------------------------
def ask_gpt(user_text: str) -> str:
    conversation.append(HumanMessage(content=user_text))

    response = llm.invoke(conversation)
    conversation.append(AIMessage(content=response.content))

    return response.content

# ---------------------------------
# Text → Speech
# ---------------------------------
def speak(text: str):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ---------------------------------
# Assistant loop
# ---------------------------------
def run_assistant():
    print("Voice assistant started")
    print("Say 'exit' or 'quit' to stop")

    while True:
        record_audio()

        print("\nTranscribing...")
        user_text = transcribe_audio(AUDIO_FILE)
        print(f"You said: {user_text}")

        if user_text.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye!")
            break

        print("\nThinking...")
        answer = ask_gpt(user_text)
        print(f"Assistant: {answer}")

        speak(answer)

# ---------------------------------
# Run
# ---------------------------------
if __name__ == "__main__":
    run_assistant()
