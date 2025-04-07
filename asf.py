import os
import re

def rename_files_in_directory(parent_directory):
    for folder_name in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder_name)

        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            for filename in os.listdir(folder_path):
                # Regex: Match (optional 'cropped_') X _ Y then anything, then extension
                match = re.match(r'^(cropped_)?(-?\d+)_(-?\d+).*?(\.\w+)$', filename)

                if match:
                    prefix = match.group(1) or ''   # 'cropped_' or ''
                    x = match.group(2)
                    y = match.group(3)
                    extension = match.group(4)

                    new_name = f"{prefix}{x}_{y}{extension}"
                    src = os.path.join(folder_path, filename)
                    dst = os.path.join(folder_path, new_name)

                    os.rename(src, dst)
                    print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    directory_path = "Media/Reward Maps/"  # Replace with your path
    rename_files_in_directory(directory_path)
