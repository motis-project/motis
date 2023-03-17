#!/usr/bin/python3

import glob
import re
import subprocess
from multiprocessing import Pool
import os
import sys

dir = os.getcwd()

routers = ["routing", "nigiri"]


def query_f(id, router):
    return f"{id}_queries_{router}.json"


def result_f(id, router):
    return f"{id}_responses_{router}.json"


def reproduce(filepath, verbose=False):
    m = re.search(r'fail/([0-9]*).*', filepath)
    id = m.group(1)

    subprocess.run(["rm", "-rf", "{}/input-{}".format(dir, id)], check=True)
    subprocess.run(["rm", "-rf", "{}/data_{}".format(dir, id)], check=True)

    run_xtract = [
        "./motis",
        "xtract",
        "input/hrd",
        "input-{}".format(id),
        f"fail/{result_f(id, routers[0])}",
        f"fail/{result_f(id, routers[1])}"
    ]
    subprocess.check_call(run_xtract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if verbose:
        print("###", " ".join(run_xtract))
        subprocess.run(run_xtract, check=True)
    else:
        subprocess.check_call(run_xtract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.check_call(["mkdir", "{}/data_{}".format(dir, id)])
    subprocess.check_call(["rm", "-rf", "{}/input-{}/stamm".format(dir, id)])
    subprocess.check_call(["rm", "-rf", "{}/data_{}/osrm".format(dir, id)])
    subprocess.check_call(["ln", "-s", "{}/input/hrd/stamm".format(dir), "input-{}/stamm".format(id)])
    subprocess.check_call(["ln", "-s", "{}/input/osm.pbf".format(dir), "input-{}/osm.pbf".format(id)])
    subprocess.check_call(["ln", "-s", "{}/data-full/osrm".format(dir), "data_{}/osrm".format(id)])

    run_routing = [
        "./motis",
        "--modules", "routing", "intermodal", "lookup", "osrm",
        "--dataset.write_serialized=false",
        "--dataset.cache_graph=false",
        "--dataset.read_graph=false",
        "--import.paths", "schedule:input-{}".format(id), "osm:input/osm.pbf".format(id),
        "--import.data_dir=data_{}".format(id),
        f"--batch_input_file=fail/{query_f(id, routers[0])}",
        f"--batch_output_file={result_f(id, routers[0])}",
        "--num_threads", "1"
    ]
    if verbose:
        print("###", " ".join(run_routing))
        subprocess.run(run_routing, check=True)
    else:
        subprocess.check_call(run_routing, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    run_nigiri = [
        "./motis",
        "--modules", "nigiri", "intermodal", "lookup", "osrm",
        "--dataset.write_serialized=false",
        "--dataset.cache_graph=false",
        "--dataset.read_graph=false",
        "--nigiri.no_cache=true",
        "--import.paths", "schedule:input-{}".format(id), "osm:input/osm.pbf",
        "--import.data_dir=data_{}".format(id),
        f"--batch_input_file=fail/{query_f(id, routers[1])}",
        f"--batch_output_file={result_f(id, routers[1])}",
        "--num_threads", "1"
    ]
    if verbose:
        print("###", " ".join(run_nigiri))
        subprocess.run(run_nigiri, check=True)
    else:
        subprocess.check_call(run_nigiri, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        run_compare = [
            "./motis",
            "intermodal_compare",
            "--fail", "",
            "--queries",
            f"fail/{query_f(id, routers[0])}",
            f"fail/{query_f(id, routers[1])}",
            "--input",
            result_f(id, routers[0]),
            result_f(id, routers[1])
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
            result_f(id, routers[0]),
            result_f(id, routers[1])
        ]))
        print("COMPARE_CMD:", " ".join(e.cmd))
        subprocess.run(e.cmd)
        print("\n\n\n\n")
        return 1

    if verbose:
        print(id, " good")

    return 0


if len(sys.argv) < 2:
    with Pool(processes=6) as pool:
        glob_str = f"fail/{query_f('*', routers[0])}"
        files = glob.iglob(glob_str)
        reproducable = pool.map(reproduce, files)
else:
    reproduce(f"fail/{query_f(sys.argv[1], routers[0])}", True)
