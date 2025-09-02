#!/usr/bin/python3

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    try:
        print(f"Running: {' '.join(cmd)}")
        if cwd:
            print(f"Working directory: {cwd}")
        
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        raise


def get_directories(path):
    """Get all directories in the given path."""
    try:
        return [d for d in os.listdir(path) if d != 'data' and os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        print(f"Directory {path} not found")
        return []


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <ID>")
        sys.exit(1)
    
    id_value = sys.argv[1]
    
    # Step 1: Execute motis extract
    print(f"\n=== Step 1: Extracting data for ID {id_value} ===")
    extract_cmd = ["./motis", "extract", "-i", f"fail/{id_value}_0.json", f"fail/{id_value}_1.json", "-o", id_value]
    
    try:
        run_command(extract_cmd)
    except subprocess.CalledProcessError:
        print(f"Failed to extract data for ID {id_value}")
        sys.exit(1)
    
    # Verify the directory was created
    if not os.path.exists(id_value):
        print(f"Error: Directory {id_value} was not created")
        sys.exit(1)


    # Step 2: Get timetable directories and run motis import
    print(f"\n=== Step 2: Importing timetables for ID {id_value} ===")
    timetable_dirs = get_directories(id_value)
    
    if not timetable_dirs:
        print(f"Warning: No timetable directories found in {id_value}")
        sys.exit(1)
    
    print(f"Found timetable directories: {timetable_dirs}")
    

    # Create config.yml
    try:
        run_command(["../motis", "config"] + timetable_dirs, cwd=id_value)
    except subprocess.CalledProcessError:
        print(f"Failed to configure timetables for ID {id_value}")
        sys.exit(1)

    # Import
    try:
        run_command(["../motis", "import"], cwd=id_value)
    except subprocess.CalledProcessError:
        print(f"Failed to import timetables for ID {id_value}")
        sys.exit(1)



    # Step 3: Copy query file
    print(f"\n=== Step 3: Copying query file for ID {id_value} ===")
    source_file = f"fail/{id_value}_q.txt"
    dest_file = f"{id_value}/queries.txt"
    
    try:
        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}")
        else:
            print(f"Warning: Source file {source_file} not found in {os.getcwd()}")
    except Exception as e:
        print(f"Error copying query file: {e}")
        sys.exit(1)
    
    print(f"\n=== Processing complete for ID {id_value} ===")


if __name__ == "__main__":
    main()