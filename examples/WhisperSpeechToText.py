import whisper

def get_reference_text(audio_path):
    # 'base' or 'small' is usually enough for a short 5-10s reference clip
    model = whisper.load_model("base") 
    
    # Transcribe the audio file
    result = model.transcribe(audio_path)
    
    # Strip leading/trailing whitespace for a clean ref_text
    return result["text"].strip()

# Example Usage
audio_file = "/data/audio/Adolf_Hitler_Speech_1939.wav"
ref_text = get_reference_text(audio_file)

print(f"Generated ref_text: {ref_text}")
