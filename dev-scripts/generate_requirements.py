import subprocess
import re

from pathlib import Path


# Define your specific Git repository and the subdirectory to handle
git_repo_url = "git+https://github.com/MuhammadUsamaSattar/FM3-MicRo.git"
editable_subdirectory = "gymnasium_envs"
editable_format = f"-e ./{editable_subdirectory}"  # Editable format for the local path

# Define the map of specific PyTorch packages and versions to their URLs
torch_url_map = {
    "torch==2.5.1+cu124": "https://download.pytorch.org/whl/cu124",
    "torchaudio==2.5.1+cu124": "https://download.pytorch.org/whl/cu124",
    "torchvision==0.20.1+cu124": "https://download.pytorch.org/whl/cu124",
}

# Path to the requirements file
requirements_file = Path(__file__).parent.parent / "requirements.txt"

# Step 1: Run pip freeze and capture the output
try:
    result = subprocess.run(
        ["pip", "freeze"], capture_output=True, text=True, check=True
    )
    frozen_requirements = result.stdout.splitlines()
except subprocess.CalledProcessError as e:
    print(f"Error running pip freeze: {e}")
    frozen_requirements = []

# Step 2: Process the output to include custom URLs and handle editable installs
processed_requirements = []
for line in frozen_requirements:
    # 2.1: Replace PyTorch-related packages with custom URLs
    match = re.match(r"(torch(?:audio|vision)?==([\d.]+)\+cu\d+)", line)
    if match:
        package_name = match.group(1)  # Full package line, e.g., torch==2.5.1+cu124
        if package_name in torch_url_map:
            # Replace with custom URL format
            line = f"{package_name} @ {torch_url_map[package_name]}"

    # 2.2: Replace the specific Git repository link with an editable install format for gymnasium_envs
    if git_repo_url in line and editable_subdirectory in line:
        # Replace with the editable install format
        line = editable_format

    processed_requirements.append(line)

# Step 3: Write the updated requirements to the file
with open(requirements_file, "w") as file:
    file.write("\n".join(processed_requirements) + "\n")

print(f"Updated requirements written to {requirements_file}")
