import json
import os
import copy # Added for data safety

class ConfigLoader:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Store metadata
        metadata = json_data.get("project_metadata", {})
        self.base_output_dir = metadata.get("file_path", "")
        self.model_path = metadata.get("model_path")
        self.audio_clips = json_data.get('audio_clips',{})

    def get_task(self, key):
        """Retrieves a copy of the task with the injected full path."""
        raw_task = self.audio_clips.get(key)
        
        if raw_task:
            # Create a deep copy so we don't accidentally modify self.clips
            task = copy.deepcopy(raw_task)
            
            # Use the provided output_file or fall back to the key name
            filename = task.get("output_file", f"{key}.wav")
            
            # Build the absolute path
            task["full_output_path"] = os.path.join(self.base_output_dir, filename)
            return task
            
        return None

    def get_all_keys(self):
        """Returns the keys available in the current clip node."""
        return list(self.audio_clips.keys())
