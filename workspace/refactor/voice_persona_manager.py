class PersonaManager:
    def __init__(self, model_container):
        self.engine = model_container
        self._cache = {}

    def get_persona(self, ref_audio, ref_text):
        # Cache by audio path to avoid re-encoding the same voice
        if ref_audio not in self._cache:
            self._cache[ref_audio] = self.engine.create_prompt(ref_audio, ref_text)
        return self._cache[ref_audio]


