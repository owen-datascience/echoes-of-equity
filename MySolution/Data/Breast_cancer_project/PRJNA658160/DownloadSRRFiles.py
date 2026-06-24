import subprocess
import os

# Path to SRA Toolkit fasterq-dump
FASTERQ_DUMP = r"C:\sratoolkit\bin\fasterq-dump.exe"

# File containing SRR IDs
SRR_FILE = "SRR_Acc_List_4.txt"

# Check if fasterq-dump exists
if not os.path.exists(FASTERQ_DUMP):
    raise FileNotFoundError("fasterq-dump.exe not found at: " + FASTERQ_DUMP)

# Get the directory where the script is actually sitting
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"1. Current Working Directory: {current_dir}")

# Construct the full path to the target folder
full_path = os.path.join(current_dir, SRR_FILE)

# Read SRR accessions
with open(full_path, "r") as f:
    srr_list = [line.strip() for line in f if line.strip()]

# Download each SRR
for srr in srr_list:
    print(f"\n=== Downloading {srr} ===\n")

    # Build the command
    cmd = [FASTERQ_DUMP, srr, "--split-files"]

    # Run the command
    process = subprocess.run(cmd, capture_output=True, text=True)

    # Print output
    print(process.stdout)
    if process.stderr:
        print("Warnings/Errors:")
        print(process.stderr)

print("\nAll downloads completed!")
