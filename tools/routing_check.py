#!/bin/python3

from enum import Enum
import subprocess

generate = False
run_routers = ["/nigiri"]
cmp_routers = ["/routing", "/nigiri"]

num_queries = 500

class StartType(Enum):
    PreTrip = 1
    OnTripStation = 2
    IntermodalPreTrip = 3
    IntermodalOnTrip = 4

class Query:
    interalmodal_dest = True
    start_type = StartType.PreTrip
    forward = True

    def cmd(self):
        cmd = [
            "./motis",
            "generate",
            "-c", "input/generate-config.ini"
        ]

        cmd.extend(["--routers"])
        cmd.extend(cmp_routers)

        cmd.extend(["--query_count", str(num_queries)])

        if self.forward:
            file_name = "fwd"
            cmd.extend(["--search_dir", "forward"])
        else:
            file_name = "bwd"
            cmd.extend(["--search_dir", "backward"])

        if self.start_type == StartType.PreTrip:
            file_name = file_name + "_pretrip"
            cmd.extend(["--start_type", "pretrip"])
        elif self.start_type == StartType.OnTripStation:
            file_name = file_name + "_ontrip"
            cmd.extend(["--start_type", "ontrip_station"])
        elif self.start_type == StartType.IntermodalPreTrip:
            file_name = file_name + "_ipretrip"
            cmd.extend(["--start_type", "intermodal_pretrip",
                        "--start_modes", "osrm_foot-15"])
        elif self.start_type == StartType.IntermodalOnTrip:
            file_name = file_name + "_iontrip"
            cmd.extend(["--start_type", "intermodal_ontrip",
                        "--start_modes", "osrm_foot-15"])
        else:
            raise "unknown start type"

        if intermodal_dest:
            file_name = file_name + "_idest"
            cmd.extend(["--dest_type", "coordinate",
                        "--dest_modes", "osrm_foot-15"])
        else:
            file_name = file_name + "_sdest"
            cmd.extend(["--dest_type", "station"])

        run_query_files = [f"queries/q_{file_name}-{r[1:]}.txt" for r in run_routers]
        run_response_files = [f"responses/r_{file_name}-{r[1:]}.txt" for r in run_routers]
        cmp_query_files = [f"queries/q_{file_name}-{r[1:]}.txt" for r in cmp_routers]
        cmp_response_files = [f"responses/r_{file_name}-{r[1:]}.txt" for r in cmp_routers]
        cmd.extend(["--out", f"queries/q_{file_name}-TARGET.txt"])

        return cmd, run_query_files, run_response_files, cmp_query_files, cmp_response_files


subprocess.check_call(["mkdir", "-p", "queries", "responses"])
for start_type in [StartType.PreTrip, StartType.OnTripStation, StartType.IntermodalOnTrip, StartType.IntermodalPreTrip]:
    for intermodal_dest in [True, False]:
        for forward in [True, False]:
            q = Query()
            q.start_type = start_type
            q.interalmodal_dest = intermodal_dest
            q.forward = forward

            generate_cmd, run_query_files, run_response_files, cmp_query_files, cmp_response_files = q.cmd()
            if generate:
                try:
                    print(" ".join(generate_cmd))
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as e:
                    print(" ".join(e.cmd))
                    print(e.output)
                    raise e

            for query_file, response_file in zip(run_query_files, run_response_files):
                motis_cmd = [
                    "./motis",
                    "-c", "input/config.ini",
                    "--batch_input_file", query_file,
                    "--batch_output_file", response_file
                ]
                print("    ", " ".join(motis_cmd))
                try:
                    subprocess.run(motis_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError as e:
                    print(" ".join(e.cmd))
                    print(e.output)
                    raise e

            compare_cmd = ["./motis", "compare"]
            compare_cmd.append("--queries")
            compare_cmd.extend(cmp_query_files)
            compare_cmd.append("--responses")
            compare_cmd.extend(cmp_response_files)
            print("      ", " ".join(compare_cmd))

            try:
                subprocess.check_call(compare_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                subprocess.run(e.cmd)
                raise e
