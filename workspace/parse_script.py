import re

# Garden at the Manor House. A flight of grey stone steps leads up to the house. The garden, an old-fashioned one, full of roses. 
# Time of year, July. Basket chairs, and a table covered with books, are set under a large yew-tree.
# [Miss Prism discovered seated at the table. Cecily is at the back watering flowers.]

def parse_script(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Using readlines to handle the file line-by-line
        lines = f.readlines()

    script_data = []
    current_char = None
    current_text = []

    for line in lines:
        clean_line = line.strip()
        
        if not clean_line:
            continue

        # Pattern: All caps name followed by a period (e.g., MISS PRISM.)
        if re.match(r'^[A-Z\s]+\.$', clean_line):
            # If we were already tracking a character, save their dialogue before switching
            if current_char and current_text:
                script_data.append({
                    "speaker": current_char, 
                    "text": " ".join(current_text)
                })
            
            # Reset for the new character
            current_char = clean_line.replace(".", "").strip().lower().replace(" ", "_")
            current_text = []
        else:
            # It's a line of dialogue; add it to the current speaker's buffer
            if current_char:
                current_text.append(clean_line)

    # Catch the very last segment
    if current_char and current_text:
        script_data.append({"speaker": current_char, "text": " ".join(current_text)})

    return script_data
