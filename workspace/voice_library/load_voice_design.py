import torch

# --- STEP 1: DESIGN THE VOICE ---
# This uses your instructions and a seed to create the 'Person'
# Note: Ensure prepare_seed() from our previous step is called here!
description = "A male voice with a staccato Chinese accent, sounding frantic and incredulous."
spk_embed = model.design_voice(instruct=description)

# --- STEP 2: SAVE THE IDENTITY ---
# We save it as a .pt (PyTorch) file. 
# This is the 'DNA' of the voice you just created.
torch.save(spk_embed, "unique_panicked_voice.pt")
print("Voice identity saved to disk.")

# --- STEP 3: REUSE LATER ---
# Next time you run the script, you don't need 'design_voice' anymore
loaded_spk_embed = torch.load("unique_panicked_voice.pt")

# Use the loaded identity to speak new text
audio = model.generate(
    text="Why are you still standing there? Run!", 
    spk_embed=loaded_spk_embed
)

# --- OPTION B: Load and Use that voice ---
# Use this in your production/test code

# Initialize the library
library = VoiceLibrary()

vocal_dna = library.load_voice("panicked_male_v1")

if vocal_dna is not None:
    model.generate(
        text="Why are you still standing there?! We have to go now!",
        spk_embed=vocal_dna
    )
