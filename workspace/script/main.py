# 1. Initialize your loader
config = ConfigLoader("config.json")

# 2. Parse the script file
script_lines = parse_script("importance_of_being_earnest.txt")

# 3. Iterate and Speak
print(f"Starting Script Processing...\n" + "="*30)

for char_id, dialogue in script_lines:
    # Fetch persona config (mapping 'miss_prism' to your JSON ID)
    task = config.get_task(char_id)
    
    if task:
        # Override the generic quote with the specific script dialogue
        task['text'] = dialogue
        
        print(f"\n[CHARACTER] {char_id.upper()}")
        print(f"[SPEAKING] {dialogue[:60]}...")
        
        # --- CALL YOUR TTS FUNCTION HERE ---
        # Example: tts_model.generate(task)
        # generate_audio(task)
        
    else:
        print(f"\n[WARNING] No persona found for ID: {char_id}")

print(f"\n" + "="*30 + "\nScript Complete.")
