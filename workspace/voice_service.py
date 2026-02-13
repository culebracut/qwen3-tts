class VoiceGenerationService:
    def __init__(self, model_container, config_manager, persona_manager):
        self.engine = model_container
        self.configs = config_manager
        self.personas = persona_manager

    def process_task(self, task, dry_run=False):
        if not task:
            return None

        # Extract the pre-merged path from our ConfigLoader
        save_path = task.get("full_output_path")

        # Actual AI execution
        persona = self.personas.get_persona(task['id'], task["ref_audio"], task["ref_text"])
        wav, sr = self.engine.generate(
            text=task["text"],
            language=task["language"],
            prompt=persona,
            instruct=task["instruct"],
            seed=task['seed'],
            temp=task['temp']
        )
        
        return {
            "wav": wav, 
            "sr": sr, 
            "path": save_path, 
            "key": task
        }

    def generate_tasks(self):
        for key in self.personas:
            yield self.process_task(key)
