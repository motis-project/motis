#!/usr/bin/python3

import os
import sys
import subprocess
import shutil
import yaml
from multiprocessing import Pool
from pathlib import Path


def resolve_motis_exec():
    env_exec = os.environ.get("MOTIS_EXEC")
    if env_exec:
        return env_exec

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "build" / "motis",
        repo_root / "cmake-build-relwithdebinfo" / "motis",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return "motis"


MOTIS_EXEC = resolve_motis_exec()

QUERIES = {
    'raptor': {
        'params': '?algorithm=RAPTOR&numItineraries=5&maxItineraries=5',
        'exec': MOTIS_EXEC
    },
    'pong': {
        'params': '?algorithm=PONG&numItineraries=5&maxItineraries=5',
        'exec': MOTIS_EXEC
    }
}


def update_timetable_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    # config['timetable']['tb'] = True
    config['timetable']['first_day'] = '2025-10-04'

    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def cmd(cmd, cwd=None, verbose=False):
    try:
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            if cwd:
                print(f"Working directory: {cwd}")

        result = subprocess.run(cmd, cwd=cwd, check=True,
                                capture_output=True,
                                text=True)

        if verbose:
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)} [CWD={cwd}]")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")


def get_directories(path):
    """Get all directories in the given path."""
    try:
        return [d for d in os.listdir(path) if
                d != 'data' and os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        print(f"Directory {path} not found")
        return []


def try_reproduce(id_value, verbose=False):
    motis = next(iter(QUERIES.items()))[1]['exec']
    dir = f"reproduce/{id_value}"
    os.makedirs(dir, exist_ok=True)

    # Step 1: Execute motis extract
    extract_cmd = [
        motis,
        "extract",
        "--filter_stops", "false",
        "-i",
        f"fail/{id_value}_0.json"
    ]

    # Add second JSON file only if it exists
    second_json = f"fail/{id_value}_1.json"
    if os.path.exists(second_json):
        extract_cmd.append(second_json)

    extract_cmd.extend(["-o", dir, '--reduce', 'true'])

    cmd(extract_cmd, verbose=verbose)

    # Step 2: Get timetable directories and run motis import
    timetable_dirs = get_directories(dir)
    if not timetable_dirs:
        print(f"Warning: No timetable directories found in {id_value}")
        sys.exit(1)

    cmd([motis, 'config'] + timetable_dirs, cwd=dir, verbose=verbose)
    update_timetable_config(f"{dir}/config.yml")
    cmd([motis, 'import'], cwd=dir, verbose=verbose)

    # Step 3: Copy query file
    for name, run in QUERIES.items():
        cmd([
            motis,
            'params',
            '-i', f"../../fail/{id_value}_q.txt",
            '-o', f"queries-{name}.txt",
            '-p', run['params']
        ], cwd=dir, verbose=verbose)

    # Step 4: Run Queries
    for name, params in QUERIES.items():
        cmd([
            params['exec'],
            'batch',
            '-q', f"queries-{name}.txt",
            '-r', f"responses-{name}.txt"
        ], cwd=dir, verbose=verbose)

    # Step 5: Compare
    compare_cmd = [
        motis,
        'compare',
        '-q', f"queries-{next(iter(QUERIES))}.txt",
        '-r'
    ]
    compare_cmd.extend([
        f"responses-{name}.txt" for name, params in QUERIES.items()
    ])

    if cmd(compare_cmd, cwd=dir, verbose=verbose) == 0:
        if verbose:
            print("NO DIFF")
        return False
    else:
        if verbose:
            print("REPRODUCED")
        return True


if __name__ == "__main__":
    if len(sys.argv) == 2:
        id_value = sys.argv[1]
        try_reproduce(id_value, True)
    else:
        with Pool(processes=6) as pool:
            query_ids = [d.removesuffix('_q.txt') for d in os.listdir('fail') if
                         d.endswith('_q.txt')]
            pool.map(try_reproduce, query_ids)
