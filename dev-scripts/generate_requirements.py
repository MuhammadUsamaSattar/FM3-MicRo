import subprocess
from pathlib import Path


# Define your specific Git repository and the subdirectory to handle
git_repo_url = "git+https://github.com/MuhammadUsamaSattar/FM3-MicRo.git"
editable_subdirectory = "gymnasium_envs"
editable_format = f"-e ./{editable_subdirectory}"  # Editable format for the local path

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

# Step 2: Process the output to handle editable installs, spinnaker-python fix, and remove flash_attn
torch_dependencies = []  # To store the PyTorch-related dependencies
other_dependencies = []  # To store all other dependencies

for line in frozen_requirements:
    # 2.1: Handle the editable install for gymnasium_envs
    if git_repo_url in line and editable_subdirectory in line:
        line = editable_format

    # 2.2: Skip the flash_attn library
    if "flash_attn" in line:
        print(f"Skipping flash_attn dependency: {line}")
        continue

    # 2.3: Skip spinnaker-python (removes it regardless of path)
    if "spinnaker-python" in line:
        print(f"Skipping spinnaker-python dependency: {line}")
        continue

    # 2.4: Separate PyTorch-related dependencies from others
    if line.startswith(("torch==", "torchaudio==", "torchvision==")):
        torch_dependencies.append(line)
    else:
        other_dependencies.append(line)

# Step 3: Write the updated requirements to the file
with open(requirements_file, "w") as file:
    # First write the torch-related dependencies with the URL at the top
    file.write("--index-url https://download.pytorch.org/whl/cu124\n")
    file.write("\n".join(torch_dependencies) + "\n")

    # Then add the fallback PyPI URL and all other dependencies
    file.write("--extra-index-url https://pypi.org/simple\n")
    file.write("\n".join(other_dependencies) + "\n")

print(f"Updated requirements written to {requirements_file}")
