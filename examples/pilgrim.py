import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Combine CAPS for stress and ellipses for the 'Duke' pause
myInputText = "LISTEN...... pilgrim. I said...... don't *ever*...... touch that saddle again."

# Refine the instruction to support the text-based cues
myInstruct = (
    "A deep, gravelly voice with authoritative grit. "
    "Pay close attention to word emphasis and use a slow, punctuated delivery."
)

# 1. Initialize the Base model (needed for cloning)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", 
    device_map="cuda:0", 
    dtype=torch.bfloat16
)

prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False
)
wavs, sr = model.generate_voice_clone(
    text=myInputText,
    language="English",
    voice_clone_prompt=prompt_items,
    instruct=myInstruct,
    temperature=0.7 # Lowering temperature slightly helps stabilize the specific rhythm
)

# --- WHISPER TEST ---
whisper_wav, sr = model.generate_voice_clone(
    text="......keep your voice down, pilgrim......",
    language="English",
    voice_clone_prompt=prompt_items,
    instruct="Speak in a low, gravelly, secretive whisper."
)

# --- SHOUT TEST ---
shout_wav, sr = model.generate_voice_clone(
    text="GET OFF MY LAND!",
    language="English",
    voice_clone_prompt=prompt_items,
    instruct="Speak with a loud, powerful, and angry commanding shout."
)

