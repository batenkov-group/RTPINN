import os
import shutil
import re

# Directories
model_all_dir = "ModelAll"
model_filtered_dir = "ModelFiltered"

# Ensure the output directory exists
os.makedirs(model_filtered_dir, exist_ok=True)

# Sample output text file containing the model names
output_file = "output.txt"  # Replace with actual file path if needed

# Read the output file and extract model names
model_names = set()
with open(output_file, "r") as f:
    for line in f:
        match = re.search(r"model name: (.+)", line)
        if match:
            model_names.add(match.group(1))

# Print extracted model names
print("Extracted Model Names:", model_names)

# Copy matching models from ModelAll to ModelFiltered
for filename in os.listdir(model_all_dir):
    model_base_name, ext = os.path.splitext(filename)

    if model_base_name in model_names and ext in [".pt", ".pth"]:  # Adjust for valid model extensions
        src_path = os.path.join(model_all_dir, filename)
        dest_path = os.path.join(model_filtered_dir, filename)

        shutil.copy(src_path, dest_path)
        print(f"Copied: {filename} -> {model_filtered_dir}")

print("Filtering process completed!")
