import whisper

model = whisper.load_model("small")
result = model.transcribe(r"C:\code\agenticai_realpage\whisper\MLKDream_64kb.mp3")

print(result["text"])
