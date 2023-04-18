#!/usr/bin/python3

import sys
import subprocess


def query_f(id, router):
    return f"{id}_q200_fwd_{router}.json"


def result_f(id, router):
    return f"{id}_r200_fwd_{router}.json"


if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} ID YYYY-MM-DD HH:MM")
else:
    id = int(sys.argv[1])
    start_time = sys.argv[2]
    print(f"debug for id={id}, time={start_time}")


    reproduce_dir = f"./reproduce/{id}"
    data_dir = f"{reproduce_dir}/data"
    input_dir = f"{reproduce_dir}/input"

    run_nigiri = [
        "./motis",
        "-c", "input/config.ini",
        "--modules", "nigiri", "intermodal", "lookup", "osrm",
        "--dataset.cache_graph=true",
        "--dataset.read_graph=false",
        "--dataset.read_graph_mmap=true",
        "--nigiri.no_cache=true",
        "--import.paths", f"schedule:{input_dir}/schedule", f"osm:input/osm.pbf",
        f"--import.data_dir={data_dir}",
        f"--batch_input_file=fail/{query_f(id, 'nigiri')}",
        f"--batch_output_file={reproduce_dir}/{result_f(id, 'nigiri')}",
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
    printed = False
    for line in out.splitlines():
        if line.startswith(needle):
            do_print = True
        elif line.startswith(b"init: "):
            do_print = False

        if do_print:
            printed = True
            print(line.decode("utf-8"))

    if not printed:
        print("NOTHING FOUND")
