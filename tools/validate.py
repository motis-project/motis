#!/usr/bin/env python3
"""Validate McRAPTOR against the RAPTOR baseline on an imported dataset.

For every case, the same generated queries are run through `motis batch`
twice - once with the reference algorithm (PONG / RAPTOR) and once with
the McRAPTOR-based algorithm (PONG_MCRAPTOR / MCRAPTOR) - and the
responses are checked for equality with `motis compare`. The debugOutput
`algorithm` field (the algorithm that actually ran, after fallbacks) is
used to detect silent fallbacks to the reference algorithm.

Example:
  python3 tools/validate.py ./build/motis --data ~/ger/data \\
    --name germany --date 2027-05-20 --intermodal --n 50
"""

import argparse
import os
import subprocess
import sys
import tempfile

ALL_MODES = ["AIRPLANE", "HIGHSPEED_RAIL", "LONG_DISTANCE", "COACH",
             "NIGHT_RAIL", "RIDE_SHARING", "REGIONAL_RAIL", "SUBURBAN",
             "SUBWAY", "TRAM", "BUS", "FERRY", "ODM", "FUNICULAR",
             "AERIAL_LIFT", "OTHER"]

# generated api::algorithmEnum values (openapi.yaml enum order)
ALGO_ENUM = {"RAPTOR": 0, "MCRAPTOR": 1, "PONG": 2, "PONG_MCRAPTOR": 3,
             "TB": 4, "MCRAPTOR_COST": 5, "PONG_MCRAPTOR_COST": 6}

PAIRS = [("PONG", "PONG_MCRAPTOR"), ("RAPTOR", "MCRAPTOR")]


def run(cmd, cwd=None, log=None):
    out = subprocess.run(cmd, cwd=cwd, check=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT).stdout.decode()
    if log:
        with open(log, "w") as f:
            f.write(out)
    return out


def generate(motis, data, n, date, work, walk):
    cmd = [motis, "generate", "-d", data, "-n", str(n), "--lb_rank", "0",
           "--first_day", date, "--last_day", date]
    if walk:
        cmd += ["-m", "WALK"]
    run(cmd, cwd=work)  # deterministic: seeded counter + fixed date
    with open(os.path.join(work, "queries.txt")) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        sys.exit("validate: 'generate' produced no queries (data=%s, walk=%s)"
                 % (data, walk))
    return lines


def timing_summary(log):
    # extract the per-query response time stats from the batch output
    with open(log) as f:
        lines = f.read().splitlines()
    out = {}
    for i, ln in enumerate(lines):
        if ln.strip() == "response_time" and i + 5 < len(lines):
            for x in lines[i + 1:i + 8]:
                x = x.strip()
                for key in ("average:", "99 quantile:", "50 quantile:"):
                    if x.startswith(key):
                        out[key.rstrip(":")] = x.split(":")[1].strip()
    return (" ".join("%s=%sms" % (k.replace(" quantile", "%"), v)
                     for k, v in out.items()) if out else "?")


def compare(motis, qlines, ref, uut, work, label):
    qf = os.path.join(work, label + ".q")
    with open(qf, "w") as f:
        f.write("\n".join(qlines) + "\n")
    files = []
    for i, resp in enumerate((ref, uut)):
        rf = "%s.%d.json" % (qf, i)
        with open(rf, "w") as f:
            f.write("\n".join(resp) + "\n")
        files.append(rf)
    return subprocess.run([motis, "compare", "-q", qf, "-r"] + files,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def suffix(case, algorithm):
    s = "&algorithm=" + algorithm
    if case["arrive_by"]:
        s += "&arriveBy=true"
    if case["clasz"]:
        s += "&transitModes=" + case["clasz"]
    if case["routed"]:
        s += "&useRoutedTransfers=true"  # prf_idx 1 = osr-routed footpaths
    return s


def build_cases(bases, restricted, routed):
    cases = []

    def add(label, base, pair, arrive_by=False, clasz=None, routed=False):
        cases.append(dict(label=label, base=base, pair=pair,
                          arrive_by=arrive_by, clasz=clasz, routed=routed))

    for qt in bases:
        for ref, uut in PAIRS:
            p = (ref, uut)
            add("%s-%s-fwd" % (qt, uut.lower()), qt, p)
            add("%s-%s-bwd" % (qt, uut.lower()), qt, p, arrive_by=True)
            if restricted:
                add("%s-%s-fwd-clasz" % (qt, uut.lower()), qt, p,
                    clasz=restricted)
            if routed:
                add("%s-%s-fwd-routed" % (qt, uut.lower()), qt, p,
                    routed=True)
    return cases


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", help="motis binary")
    ap.add_argument("--data", required=True, help="imported data dir")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--intermodal", action="store_true",
                    help="also test -m WALK queries")
    ap.add_argument("--routed-footpaths", action="store_true",
                    help="also test useRoutedTransfers=true; requires "
                         "osr_footpath: true in the imported data")
    ap.add_argument("--exclude-transit-modes",
                    help="API transit modes dropped for the clasz-filter "
                         "cases, comma-separated (e.g. HIGHSPEED_RAIL)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--date", required=True, help="pinned query day")
    a = ap.parse_args()

    motis = os.path.abspath(a.binary)
    data = os.path.abspath(a.data)
    work = tempfile.mkdtemp(prefix="validate-%s-" % a.name)
    print("work dir: %s" % work)
    excluded = (set(a.exclude_transit_modes.split(","))
                if a.exclude_transit_modes else set())
    restricted = (",".join(m for m in ALL_MODES if m not in excluded)
                  if excluded else None)

    bases = {"station": generate(motis, data, a.n, a.date, work, walk=False)}
    if a.intermodal:
        bases["intermodal"] = generate(motis, data, a.n, a.date, work,
                                       walk=True)
    cases = build_cases(bases, restricted, a.routed_footpaths)

    # one combined batch per side (reference / uut), same query order
    spans, offset = [], 0
    ref_q = os.path.join(work, "ref.q")
    uut_q = os.path.join(work, "uut.q")
    with open(ref_q, "w") as ref_out, open(uut_q, "w") as uut_out:
        for c in cases:
            ref_lines = [q + suffix(c, c["pair"][0]) for q in bases[c["base"]]]
            uut_lines = [q + suffix(c, c["pair"][1]) for q in bases[c["base"]]]
            ref_out.write("\n".join(ref_lines) + "\n")
            uut_out.write("\n".join(uut_lines) + "\n")
            spans.append((c, ref_lines, offset, len(ref_lines)))
            offset += len(ref_lines)

    outs, timings = [], []
    for side, qfile in (("ref", ref_q), ("uut", uut_q)):
        o = qfile + ".responses"
        log = qfile + ".log"
        run([motis, "batch", "-d", data, "-q", qfile, "-r", o], cwd=work,
            log=log)
        with open(o) as f:
            outs.append(f.read().splitlines())
        timings.append(timing_summary(log))

    results, fails = [], 0
    for c, qlines, start, count in spans:
        ref = outs[0][start:start + count]
        uut = outs[1][start:start + count]
        want = '"algorithm":%d' % ALGO_ENUM[c["pair"][1]]
        fell_back = sum(1 for ln in uut
                        if '"algorithm":' in ln and want not in ln)
        if not compare(motis, qlines, ref, uut, work, c["label"]):
            results.append((c["label"], False, "mismatch (motis compare)"))
        elif fell_back:
            results.append((c["label"], False,
                            "fell back on %d/%d queries" % (fell_back, count)))
        else:
            results.append((c["label"], True, "%d ok" % count))

    print("== %s ==" % a.name)
    for label, ok, detail in results:
        print("%s %-38s %s" % ("PASS" if ok else "FAIL", label, detail))
        fails += 0 if ok else 1
    print("batch %s: %s" % (PAIRS[0][0] + "/" + PAIRS[1][0], timings[0]))
    print("batch %s: %s" % (PAIRS[0][1] + "/" + PAIRS[1][1], timings[1]))
    print("%s: %d passed, %d failed" % (a.name, len(results) - fails, fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
