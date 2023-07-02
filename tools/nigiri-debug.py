#!/usr/bin/python3

import sys
import subprocess
import os


current_dir = os.getcwd()

def query_f(id, router):
    return f"{id}_q_bwd_iontrip_idest-{router}.json"


def result_f(id, router):
    return f"{id}_r_bwd_iontrip_idest-{router}.json"


if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} ID YYYY-MM-DD HH:MM")
else:
    id = int(sys.argv[1])
    start_time = sys.argv[2]
    print(f"debug for id={id}, time={start_time}")

    reproduce_dir = f"{current_dir}/reproduce/{id}"
    fail_dir = f"{current_dir}/fail"

    run_nigiri = [
        f"{current_dir}/motis",
        "-c", "input/config.ini",
        f"--batch_input_file={fail_dir}/{query_f(id, 'nigiri')}",
        f"--batch_output_file={result_f(id, 'nigiri')}",
        "--num_threads", "1"
    ]
    print("NIGIRI_CMD:", " ".join(run_nigiri))

    out = subprocess.check_output(run_nigiri, cwd=reproduce_dir)

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
