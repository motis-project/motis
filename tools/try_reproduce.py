#!/usr/bin/python3

import glob
import re
import subprocess
from multiprocessing import Pool
import os
import sys

current_dir = os.getcwd()

routers = ["routing", "nigiri"]


def query_f(id, router):
    return f"{id}_q_{router}.json"


def result_f(id, router):
    return f"{id}_r_{router}.json"


def reproduce(filepath, verbose=False):
    m = re.search(r'fail/([0-9]*).*', filepath)
    id = m.group(1)

    reproduce_dir = f"./reproduce/{id}"
    data_dir = f"{reproduce_dir}/data"
    input_dir = f"{reproduce_dir}/input"
    fail_dir = f"{current_dir}/fail"
    subprocess.check_call(["rm", "-rf", reproduce_dir])
    subprocess.check_call(["mkdir", "-p", data_dir])
    subprocess.check_call(["mkdir", "-p", input_dir])

    if False and verbose:
        run_rewrite = [
            "./motis",
            "rewrite",
            "--in", f"{fail_dir}/{result_f(id, routers[1])}",
            "--out", f"{reproduce_dir}/check_orig_{result_f(id, routers[1])}",
            "--target", "/cc"
        ]
        print("###", " ".join(run_rewrite))
        subprocess.run(run_rewrite, check=True)

        try:
            run_check = [
                "./motis",
                "-c", "input/config.ini",
                "--modules", "cc",
                "--mode", "init",
                "--init", f"{reproduce_dir}/check_orig_{result_f(id, routers[1])}"
            ]
            print("###", " ".join(run_check))
            subprocess.run(run_check, check=True)
        except subprocess.CalledProcessError as e:
            print("NOT FEASILBE")


    if verbose:
        print("extracting...")
    run_xtract = [
        "./motis",
        "xtract",
        "-c", "input/config.ini",
        "--new_schedule", f"{input_dir}/schedule",
        "--responses",
        f"{fail_dir}/{result_f(id, routers[0])}",
        f"{fail_dir}/{result_f(id, routers[1])}"
    ]
    if verbose:
        print("###", " ".join(run_xtract))
        subprocess.run(run_xtract, check=True)
    else:
        subprocess.check_call(run_xtract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if verbose:
        print("extracting...")
    subprocess.check_call(["ln", "-s", f"{current_dir}/input/osm.pbf", input_dir])
    subprocess.check_call(["ln", "-s", f"{current_dir}/data/osrm", data_dir])

    try:
        run_routing = [
            f"{current_dir}/motis",
            "-c", "input/config.ini",
            f"--batch_input_file={fail_dir}/{query_f(id, routers[0])}",
            f"--batch_output_file={result_f(id, routers[0])}",
            "--num_threads", "1"
        ]
        if verbose:
            print("###", "cd", reproduce_dir, "&&", " ".join(run_routing))
            subprocess.run(run_routing, check=True, cwd=reproduce_dir)
        else:
            subprocess.check_call(
                run_routing,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=reproduce_dir)

        run_nigiri = [
            f"{current_dir}/motis",
            "-c", "input/config.ini",
            f"--batch_input_file={fail_dir}/{query_f(id, routers[1])}",
            f"--batch_output_file={result_f(id, routers[1])}",
            "--num_threads", "1"
        ]
        if verbose:
            print("###", "cd", reproduce_dir, "&&", " ".join(run_nigiri))
            subprocess.run(run_nigiri, check=True, cwd=reproduce_dir)
        else:
            subprocess.check_call(
                run_nigiri,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=reproduce_dir)
    except subprocess.CalledProcessError as e:
        print(f"FAIL: {' '.join(e.cmd)}")
        print(e.output)
        raise e

    try:
        run_compare = [
            "./motis",
            "compare",
            "--fail", "",
            "--queries",
            f"fail/{query_f(id, routers[0])}",
            f"fail/{query_f(id, routers[1])}",
            "--responses",
            f"{reproduce_dir}/{result_f(id, routers[0])}",
            f"{reproduce_dir}/{result_f(id, routers[1])}"
        ]
        if verbose:
            print("###", " ".join(run_compare))
            subprocess.run(run_compare, check=True)
        else:
            subprocess.check_call(run_compare, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("ROUTING_CMD:", " ".join(run_routing))
        print("NIGIRI_CMD:", " ".join(run_nigiri))
        print("CONNECTIONS:", " ".join([
            "./motis",
            "print",
            f"{reproduce_dir}/{result_f(id, routers[0])}",
            f"{reproduce_dir}/{result_f(id, routers[1])}"
        ]))
        print("COMPARE_CMD:", " ".join(e.cmd))
        subprocess.run(e.cmd)

        if verbose:
            run_rewrite = [
                "./motis",
                "rewrite",
                "--in", f"{reproduce_dir}/{result_f(id, routers[0])}",
                "--out", f"{reproduce_dir}/check_{result_f(id, routers[0])}",
                "--target", "/cc"
            ]
            print("###", " ".join(run_rewrite))
            subprocess.run(run_rewrite, check=True)

            run_rewrite = [
                "./motis",
                "rewrite",
                "--in", f"{reproduce_dir}/{result_f(id, routers[1])}",
                "--out", f"{reproduce_dir}/check_{result_f(id, routers[1])}",
                "--target", "/cc"
            ]
            print("###", " ".join(run_rewrite))
            subprocess.run(run_rewrite, check=True)

            run_check = [
                f"{current_dir}/motis",
                "-c", "input/config.ini",
                "--modules", "cc",
                "--mode", "init",
                "--init", f"./check_{result_f(id, routers[0])}"
            ]
            print("###", " ".join(run_check))
            subprocess.run(run_check, check=False, cwd=reproduce_dir)

            run_check = [
                f"{current_dir}/motis",
                "-c", "input/config.ini",
                "--modules", "cc",
                "--mode", "init",
                "--init", f"./check_{result_f(id, routers[1])}"
            ]
            print("###", " ".join(run_check))
            subprocess.run(run_check, check=False, cwd=reproduce_dir)

        print("\n\n\n\n")
        return 1

    if verbose:
        print(id, " good")

    return 0


if len(sys.argv) < 2:
    with Pool(processes=16) as pool:
        glob_str = f"fail/{query_f('*', routers[0])}"
        files = glob.iglob(glob_str)
        reproducable = pool.map(reproduce, files)
else:
    reproduce(f"fail/{query_f(sys.argv[1], routers[0])}", True)
