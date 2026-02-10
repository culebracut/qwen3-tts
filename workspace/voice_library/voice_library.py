import torch
import os

class VoiceLibrary:
    def __init__(self, library_path="voice_library"):
        self.library_path = library_path
        if not os.path.exists(self.library_path):
            os.makedirs(self.library_path)

    def save_voice(self, spk_embed, name):
        """Saves a designed speaker embedding to disk."""
        file_path = os.path.join(self.library_path, f"{name}.pt")
        torch.save(spk_embed, file_path)
        print(f"Successfully saved voice: '{name}' to {file_path}")

    def load_voice(self, name):
        """Loads a saved speaker embedding by name."""
        file_path = os.path.join(self.library_path, f"{name}.pt")
        if os.path.exists(file_path):
            print(f"Loading voice identity: '{name}'")
            return torch.load(file_path)
        else:
            print(f"Error: Voice '{name}' not found in library.")
            return None

    def list_voices(self):
        """Returns a list of all saved voice names."""
        return [f.replace(".pt", "") for f in os.listdir(self.library_path) if f.endswith(".pt")]
