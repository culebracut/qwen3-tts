class PersonaManager:
    def __init__(self, model_container):
        self.engine = model_container
        self._cache = {}

    def get_persona(self, ref_audio, ref_text):
        # Using ref_audio path as the unique cache key
        if ref_audio in self._cache:
            return self._cache[ref_audio]

        prompt = self.engine.create_prompt(ref_audio, ref_text)
        self._cache[ref_audio] = prompt
        return prompt
