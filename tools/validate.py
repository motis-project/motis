#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile

ALL_MODES = ["AIRPLANE", "HIGHSPEED_RAIL", "LONG_DISTANCE", "COACH",
             "NIGHT_RAIL", "RIDE_SHARING", "REGIONAL_RAIL", "SUBURBAN",
             "SUBWAY", "TRAM", "BUS", "FERRY", "ODM", "FUNICULAR",
             "AERIAL_LIFT", "OTHER"]

def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate(motis, data, n, date, work, walk):
    cmd = [motis, "generate", "-d", data, "-n", str(n), "--lb_rank", "0",
           "--first_day", date, "--last_day", date]
    if walk:
        cmd += ["-m", "WALK"]
    run(cmd, cwd=work)  # deterministic: seeded counter + fixed date
    with open(os.path.join(work, "queries.txt")) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        sys.exit("validate: 'generate' produced no queries (data=%s, walk=%s)" % (data, walk))
    return lines


def batch(motis, data, qfile, out, rt_dir):
    cmd = [motis, "batch", "-d", data, "-q", qfile, "-r", out]
    if rt_dir:
        cmd.append("--rt")  # applies dump_rt/ from cwd
    run(cmd, cwd=rt_dir)


def compare(motis, qlines, responses, work, label):
    # Write queries.
    qf = os.path.join(work, label + ".q")
    with open(qf, "w") as f:
        f.write("\n".join(qlines) + "\n")
        
    # Write responses.
    files = []
    for i, resp in enumerate(responses):
        rf = "%s.%d.json" % (qf, i)
        with open(rf, "w") as f:
            f.write("\n".join(resp) + "\n")
        files.append(rf)
        
    # Run compare for reference vs all.
    return all(subprocess.run([motis, "compare", "-q", qf, "-r", files[0], f],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
               for f in files[1:])


def suffix(case):
    s = "&algorithm=" + case["algorithm"]
    if case["arrive_by"]:
        s += "&arriveBy=true"
    if case["clasz"]:
        s += "&transitModes=" + case["clasz"]
    return s


def build_cases(bases, restricted, rt):
    cases = []

    def add(label, base, algorithm="PONG", arrive_by=False, clasz=None, rt=False):
        cases.append(dict(label=label, base=base, algorithm=algorithm,
                          arrive_by=arrive_by, clasz=clasz, rt=rt))

    for qt in bases:
        for algo in ("PONG", "RAPTOR"):
            add("%s-%s-fwd" % (qt, algo.lower()), qt, algorithm=algo)
            add("%s-%s-bwd" % (qt, algo.lower()), qt, algorithm=algo, arrive_by=True)
        if restricted:
            add("%s-pong-fwd-clasz" % qt, qt, clasz=restricted)
            add("%s-pong-bwd-clasz" % qt, qt, arrive_by=True, clasz=restricted)
    if rt:
        add("station-pong-fwd-rt", "station", rt=True)
        add("station-pong-bwd-rt", "station", arrive_by=True, rt=True)
        add("station-raptor-fwd-rt", "station", algorithm="RAPTOR", rt=True)
        if restricted:
            add("station-pong-fwd-clasz-rt", "station", clasz=restricted, rt=True)
            add("station-pong-bwd-clasz-rt", "station", arrive_by=True, clasz=restricted, rt=True)
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binaries", nargs="+", help="motis binaries to compare (first = reference)")
    ap.add_argument("--data", required=True, help="imported data dir (tt + osr)")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--rt-dir", help="dir containing dump_rt/ (enables --rt cases)")
    ap.add_argument("--intermodal", action="store_true", help="also test -m WALK queries")
    ap.add_argument("--exclude-transit-modes",
                    help="API transit modes dropped for the clasz-filter cases, "
                         "comma-separated (e.g. COACH or HIGHSPEED_RAIL,COACH)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--date", required=True, help="pinned query day (must match the rt dump)")
    a = ap.parse_args()

    bins = [os.path.abspath(b) for b in a.binaries]
    data = os.path.abspath(a.data)
    rt_dir = os.path.abspath(a.rt_dir) if a.rt_dir else None
    work = tempfile.mkdtemp(prefix="validate-%s-" % a.name)
    excluded = set(a.exclude_transit_modes.split(",")) if a.exclude_transit_modes else set()
    restricted = ",".join(m for m in ALL_MODES if m not in excluded) if excluded else None

    bases = {"station": generate(bins[0], data, a.n, a.date, work, walk=False)}
    if a.intermodal:
        bases["intermodal"] = generate(bins[0], data, a.n, a.date, work, walk=True)
    cases = build_cases(bases, restricted, rt_dir is not None)

    results = []
    for rt in (False, True):
        # FILTER CASES
        group = [c for c in cases if c["rt"] == rt]
        if not group:
            continue

        # WRITE QUERIES BATCH
        combined = os.path.join(work, "rt%d.q" % rt)
        spans, offsets = [], 0
        with open(combined, "w") as out:
            for c in group:
                qlines = [q + suffix(c) for q in bases[c["base"]]]
                out.write("\n".join(qlines) + "\n")
                spans.append((c["label"], qlines, offsets, len(qlines)))
                offsets += len(qlines)
  
        # RUN BATCH
        outs = []
        for i, b in enumerate(bins):
            o = "%s.%d" % (combined, i)
            batch(b, data, combined, o, rt_dir if rt else None)
            with open(o) as f:
                outs.append(f.read().splitlines())

        # COMPARE REF VS ALL
        for label, qlines, start, count in spans:
            responses = [o[start:start + count] for o in outs]
            gpu = max(sum(1 for ln in r if '"gpu_used":1' in ln) for r in responses)
            fell_back = next(((i, sum(1 for ln in r if '"gpu_used":0' in ln))
                              for i, r in enumerate(responses)
                              if any('"gpu_used":0' in ln for ln in r)), None)
            if not compare(bins[0], qlines, responses, work, label):
                results.append((label, False, "mismatch (motis compare)"))
            elif fell_back is not None:
                results.append((label, False, "bin%d fell back to CPU on %d/%d queries"
                                % (fell_back[0], fell_back[1], count)))
            else:
                results.append((label, True, "%d ok, gpu_used=%d" % (count, gpu)))

    if not results:
        sys.exit("validate: no cases ran -- nothing validated")

    print("== %s ==" % a.name)
    fails = 0
    for label, ok, detail in results:
        print("%s %-34s %s" % ("PASS" if ok else "FAIL", label, detail))
        fails += 0 if ok else 1
    print("%s: %d passed, %d failed" % (a.name, len(results) - fails, fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
