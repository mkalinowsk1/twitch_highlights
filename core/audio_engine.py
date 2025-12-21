# core/audio_engine.py
import whisper

def transcribe_and_search(audio_path, keywords=["wow", "amazing", "gol"]):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    
    important_segments = []
    for segment in result['segments']:
        if any(word in segment['text'].lower() for word in keywords):
            important_segments.append((segment['start'], segment['end']))
    return important_segments