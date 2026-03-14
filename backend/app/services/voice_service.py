import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")  # small & laptop-friendly

def transcribe_audio(audio_file_path):
    result = model.transcribe(audio_file_path)
    return result["text"]

print(transcribe_audio("backend/app/services/suitcase.mp3"))