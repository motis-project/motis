#!/usr/bin/python3

import os
import sys
import subprocess
import shutil
from pathlib import Path

QUERIES = {
    'raptor': {
        'params': '?algorithm=RAPTOR',
        'exec': '/home/felix/code/motis/cmake-build-release/motis'
    },
    'tb': {
        'params': '?algorithm=TB',
        'exec': '/home/felix/code/motis/cmake-build-release/motis'
    }
}


def run_command(cmd, cwd=None):
    """Run a command and handle errors."""
    try:
        print(cmd)
        print(f"Running: {' '.join(cmd)}")
        if cwd:
            print(f"Working directory: {cwd}")

        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True,
                                text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode
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
        return [d for d in os.listdir(path) if
                d != 'data' and os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        print(f"Directory {path} not found")
        return []


def run(id_value):
    motis = next(iter(QUERIES.items()))[1]['exec']
    print(motis)
    dir = f"reproduce/{id_value}"
    os.makedirs(dir, exist_ok=True)

    # Step 1: Execute motis extract
    print(f"\n=== Step 1: Extracting data for ID {id_value} ===")
    run_command([
        motis,
        "extract",
        "-i",
        f"fail/{id_value}_0.json",
        f"fail/{id_value}_1.json",
        "-o", dir
    ])

    # Step 2: Get timetable directories and run motis import
    print(f"\n=== Step 2: Importing timetables for ID {id_value} ===")
    timetable_dirs = get_directories(dir)
    if not timetable_dirs:
        print(f"Warning: No timetable directories found in {id_value}")
        sys.exit(1)

    print(f"Found timetable directories: {timetable_dirs}")
    run_command([motis, 'config'] + timetable_dirs, cwd=dir)
    run_command([motis, 'import'], cwd=dir)

    # Step 3: Copy query file
    print(f"\n=== Step 3: Copying query file for all algorithms {id_value} ===")
    for name, run in QUERIES.items():
        run_command([
            motis,
            'params',
            '-i' f"../../fail/{id_value}_q.txt",
            '-o', f"queries-{name}.txt",
            '-p', run['params']
        ], cwd=dir)

    # Step 4: Run Queries
    print(f"\n=== Step 4: Run queries {id_value} ===")
    for name, params in QUERIES.items():
        print(f"Running {name}")
        run_command([
            params['exec'],
            'batch',
            '-q', f"queries-{name}.txt",
            '-r', f"responses-{name}.txt"
        ], cwd=dir)

    # Step 5: Compare
    print(f"\n=== Step 5: Compare {id_value} ===")
    cmd = [
        motis,
        'compare',
        '-q', f"queries-{next(iter(QUERIES))}.txt",
        '-r'
    ]
    cmd.extend([
        f"responses-{name}.txt" for name, params in QUERIES.items()
    ])

    if run_command(cmd, cwd=dir) == 0:
        print("NO DIFF")
        return False
    else:
        print("REPRODUCED")
        return True


if __name__ == "__main__":
    if len(sys.argv) == 2:
        id_value = sys.argv[1]
        run(id_value)
    else:
        for q in [d.removesuffix('_q.txt') for d in os.listdir('fail') if
                  d.endswith('_q.txt')]:
            if run(q):
                break
