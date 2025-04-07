import subprocess
from pathlib import Path


# Define the git log command
git_log_command = ["git", "log", "--pretty=- %Cblue**%h:**%Creset %s"]

# Define the output file
output_file = Path(__file__).parent.parent / "CHANGELOG.md"

try:
    # Run the git log command and capture the output
    result = subprocess.run(git_log_command, capture_output=True, text=True, check=True)

    # Write the output to CHANGELONG.md
    with open(output_file, "w") as file:
        file.write(result.stdout)

    print(f"Changelog has been written to {output_file}.")
except subprocess.CalledProcessError as e:
    print(f"Error running git log: {e}")
