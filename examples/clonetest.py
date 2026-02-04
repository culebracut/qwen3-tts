import torch
import torchaudio
from qwen_tts import Qwen3TTSModel 

# 1. Setup Model
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base" # Remove trailing slash
#device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model - using 'device' is safer for single-GPU TTS wrappers
tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
) #.eval() # Always ensure .eval() for inference

# 2. Define IO
outputPath = "myWavOutputs/JohnWayne/clone/TalkThinkTooMuch.wav"
ref_audio_path = "/data/audio/JohnWayne/YouTalkThinkTooMuch.wav"  # [Your reference audio path]
ref_text_str = "You talk to0 much, think too much" # [Your full transcript]
target_text = "[very angry, shouting, historical speech style] I think, you'd better...water my horse"

# 3. Generate
with torch.no_grad():
    # Wrap in lists to ensure [1, 1] batch alignment
    wavs, sr = tts.generate_voice_clone(
        text=[target_text], 
        ref_audio=[ref_audio_path], 
        ref_text=[ref_text_str],
        language="english",
    )

# 4. Save - wavs is a list, so we take the first element [0]
# Use .float() because torchaudio.save often struggles with bfloat16 directly
""" torchaudio.save("output_english_clone.wav", wavs[0].cpu().float(), sr)

print("Successfully generated English speech!") """
# 4. Save
# Convert NumPy array to Torch tensor
output_tensor = torch.from_numpy(wavs[0]) if isinstance(wavs, list) else torch.from_numpy(wavs)

# Ensure shape is [1, T] if it's currently [T]
if output_tensor.ndim == 1:
    output_tensor = output_tensor.unsqueeze(0)

# Save (NumPy is usually float32/float64 already)
torchaudio.save(outputPath, output_tensor.to(torch.float32), sr)

print("Successfully generated English speech!")

