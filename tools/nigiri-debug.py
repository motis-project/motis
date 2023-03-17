#!/usr/bin/python3

import sys
import subprocess

def query_f(id, router):
    return f"{id}_queries_{router}.json"

def result_f(id, router):
    return f"{id}_responses_{router}.json"

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} ID START_TIME")
else:
    id = int(sys.argv[1])
    start_time = sys.argv[2]
    print(f"debug for id={id}, time={start_time}")

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
    # run_nigiri = [
    #     "./motis",
    #     "--modules", "nigiri", "intermodal", "lookup", "osrm",
    #     f"--batch_input_file=fail/{query_f(id, 'nigiri')}",
    #     f"--batch_output_file={result_f(id, 'nigiri')}",
    #     "--num_threads", "1"
    # ]
    print("NIGIRI_CMD:", " ".join(run_nigiri))

    out = subprocess.check_output(run_nigiri)

    needle = bytes("init: time_at_start={}".format(start_time), encoding='utf8')
    do_print = False
    for line in out.splitlines():
        if line.startswith(needle):
            do_print = True
        elif line.startswith(b"init: "):
            do_print = False

        if do_print:
            print(line.decode("utf-8"))
