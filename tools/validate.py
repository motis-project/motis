#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ALL_MODES = ["AIRPLANE", "HIGHSPEED_RAIL", "LONG_DISTANCE", "COACH",
             "NIGHT_RAIL", "RIDE_SHARING", "REGIONAL_RAIL", "SUBURBAN",
             "SUBWAY", "TRAM", "BUS", "FERRY", "ODM", "FUNICULAR",
             "AERIAL_LIFT", "OTHER"]

def run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate(motis, data, n, date, work, modes=None):
    cmd = [motis, "generate", "-d", data, "-n", str(n), "--lb_rank", "0",
           "--first_day", date, "--last_day", date]
    if modes:
        cmd += ["-m", modes]
    run(cmd, cwd=work)  # deterministic: seeded counter + fixed date
    with open(os.path.join(work, "queries.txt")) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        sys.exit("validate: 'generate' produced no queries (data=%s, modes=%s)" % (data, modes))
    return lines


def batch(motis, data, qfile, out, rt_dir):
    cmd = [motis, "batch", "-d", data, "-q", qfile, "-r", out]
    if rt_dir:
        cmd.append("--rt")  # applies dump_rt/ from cwd
    t0 = time.perf_counter()
    run(cmd, cwd=rt_dir)
    return time.perf_counter() - t0  # wall time (whole batch, all cases)


def pct(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def exec_times(lines):
    # per-query routing time (ms) from debugOutput.execute_time
    out = []
    for ln in lines:
        if '"execute_time"' not in ln:
            continue
        try:
            v = json.loads(ln).get("debugOutput", {}).get("execute_time")
        except (ValueError, AttributeError):
            continue
        if v is not None:
            out.append(float(v))
    return out


def bin_labels(bins):
    # parent dir name (build/cpu -> "cpu", build/cuda -> "cuda"); unique-ified
    labels, seen = [], {}
    for i, b in enumerate(bins):
        lbl = os.path.basename(os.path.dirname(b)) or ("bin%d" % i)
        if lbl in seen:
            lbl = "%s#%d" % (lbl, i)
        seen[lbl] = True
        labels.append(lbl)
    return labels


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
    if case["routed"]:
        s += "&useRoutedTransfers=true"  # prf_idx 1 = osr-routed foot footpaths
    if case["wheelchair"]:
        # prf_idx 2 + stop accessibility + td footpaths (elevator status)
        s += "&useRoutedTransfers=true&pedestrianProfile=WHEELCHAIR"
    if case["bike"]:
        s += "&requireBikeTransport=true"  # bike carriage on all transit legs
    if case["car"]:
        s += "&requireCarTransport=true"  # car carriage on all transit legs
    if case["tts"]:
        # non-default transfer time settings: additional + max(min, dur*factor)
        s += "&minTransferTime=10&transferTimeFactor=1.5&additionalTransferTime=3"
    return s


def build_cases(bases, restricted, rt, routed, wheelchair, bike, car, tts):
    cases = []

    def add(label, base, algorithm="PONG", arrive_by=False, clasz=None,
            rt=False, routed=False, wheelchair=False, bike=False, car=False,
            tts=False):
        cases.append(dict(label=label, base=base, algorithm=algorithm,
                          arrive_by=arrive_by, clasz=clasz, rt=rt,
                          routed=routed, wheelchair=wheelchair, bike=bike,
                          car=car, tts=tts))

    for qt in bases:
        if qt == "flex":
            # flex first/last mile -> td_start_/td_dest_ (in-loop td egress)
            add("flex-pong-fwd", qt)
            add("flex-pong-bwd", qt, arrive_by=True)
            add("flex-raptor-fwd", qt, algorithm="RAPTOR")
            add("flex-raptor-bwd", qt, algorithm="RAPTOR", arrive_by=True)
            continue
        for algo in ("PONG", "RAPTOR"):
            add("%s-%s-fwd" % (qt, algo.lower()), qt, algorithm=algo)
            add("%s-%s-bwd" % (qt, algo.lower()), qt, algorithm=algo, arrive_by=True)
        if restricted:
            add("%s-pong-fwd-clasz" % qt, qt, clasz=restricted)
            add("%s-pong-bwd-clasz" % qt, qt, arrive_by=True, clasz=restricted)
        if routed:
            add("%s-pong-fwd-routed" % qt, qt, routed=True)
            add("%s-pong-bwd-routed" % qt, qt, arrive_by=True, routed=True)
            add("%s-raptor-fwd-routed" % qt, qt, algorithm="RAPTOR", routed=True)
        if wheelchair:
            add("%s-pong-fwd-wheelchair" % qt, qt, wheelchair=True)
            add("%s-pong-bwd-wheelchair" % qt, qt, arrive_by=True, wheelchair=True)
        if bike:
            add("%s-pong-fwd-bike" % qt, qt, bike=True)
            add("%s-pong-bwd-bike" % qt, qt, arrive_by=True, bike=True)
            add("%s-raptor-fwd-bike" % qt, qt, algorithm="RAPTOR", bike=True)
        if car:
            add("%s-pong-fwd-car" % qt, qt, car=True)
            add("%s-pong-bwd-car" % qt, qt, arrive_by=True, car=True)
        if tts:
            add("%s-pong-fwd-tts" % qt, qt, tts=True)
            add("%s-pong-bwd-tts" % qt, qt, arrive_by=True, tts=True)
            add("%s-raptor-fwd-tts" % qt, qt, algorithm="RAPTOR", tts=True)
    if rt:
        add("station-pong-fwd-rt", "station", rt=True)
        add("station-pong-bwd-rt", "station", arrive_by=True, rt=True)
        add("station-raptor-fwd-rt", "station", algorithm="RAPTOR", rt=True)
        if restricted:
            add("station-pong-fwd-clasz-rt", "station", clasz=restricted, rt=True)
            add("station-pong-bwd-clasz-rt", "station", arrive_by=True, clasz=restricted, rt=True)
        if wheelchair:
            add("station-pong-fwd-wheelchair-rt", "station", wheelchair=True, rt=True)
        if tts:
            add("station-pong-fwd-tts-rt", "station", tts=True, rt=True)
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binaries", nargs="+", help="motis binaries to compare (first = reference)")
    ap.add_argument("--data", required=True, help="imported data dir (tt + osr)")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--rt-dir", help="dir containing dump_rt/ (enables --rt cases)")
    ap.add_argument("--intermodal", action="store_true", help="also test -m WALK queries")
    ap.add_argument("--routed-footpaths", action="store_true",
                    help="also test useRoutedTransfers=true (osr-routed foot profile); "
                         "requires osr_footpath: true in the imported data")
    ap.add_argument("--wheelchair", action="store_true",
                    help="also test pedestrianProfile=WHEELCHAIR (prf_idx 2, stop "
                         "accessibility, td footpaths); requires osr_footpath: true")
    ap.add_argument("--flex", action="store_true",
                    help="also test -m WALK,FLEX queries seeded inside flex areas "
                         "(td_start_/td_dest_ offsets); requires flex feeds + osr")
    ap.add_argument("--bike", action="store_true",
                    help="also test requireBikeTransport=true (bike carriage "
                         "route/section filters)")
    ap.add_argument("--car", action="store_true",
                    help="also test requireCarTransport=true (car carriage "
                         "route/section filters)")
    ap.add_argument("--tts", action="store_true",
                    help="also test non-default transfer time settings "
                         "(minTransferTime/transferTimeFactor/additionalTransferTime)")
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

    # CI captures a pipe, not a TTY -> Python block-buffers stdout and nothing
    # shows until exit. Line-buffer so progress is visible live.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print("== %s ==" % a.name)
    labels = bin_labels(bins)

    bases = {}
    print("generating station queries (n=%d)..." % a.n)
    bases["station"] = generate(bins[0], data, a.n, a.date, work)
    if a.intermodal:
        print("generating intermodal queries (n=%d)..." % a.n)
        bases["intermodal"] = generate(bins[0], data, a.n, a.date, work, modes="WALK")
    if a.flex:
        print("generating flex queries (n=%d)..." % a.n)
        bases["flex"] = generate(bins[0], data, a.n, a.date, work, modes="WALK,FLEX")
    cases = build_cases(bases, restricted, rt_dir is not None, a.routed_footpaths,
                        a.wheelchair, a.bike, a.car, a.tts)

    results = []
    lat = []  # (case_label, per-binary [execute_time ms] lists)
    wall = [0.0] * len(bins)  # total batch wall time per binary
    n_queries = 0  # total queries run per binary (same file for all)
    fails = 0
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
        n_queries += offsets

        # RUN BATCH (the long pole: each cold-loads tt+osr, then routes)
        outs = []
        for i, b in enumerate(bins):
            o = "%s.%d" % (combined, i)
            print("[rt=%d] batching %d queries on %s..." %
                  (rt, offsets, labels[i]))
            wall[i] += batch(b, data, combined, o, rt_dir if rt else None)
            print("[rt=%d]   %s done in %.0fs" % (rt, labels[i], wall[i]))
            with open(o) as f:
                outs.append(f.read().splitlines())

        # COMPARE REF VS ALL (print each verdict as it completes)
        for label, qlines, start, count in spans:
            responses = [o[start:start + count] for o in outs]
            lat.append((label, [exec_times(r) for r in responses]))
            gpu = max(sum(1 for ln in r if '"gpu_used":1' in ln) for r in responses)
            fell_back = next(((i, sum(1 for ln in r if '"gpu_used":0' in ln))
                              for i, r in enumerate(responses)
                              if any('"gpu_used":0' in ln for ln in r)), None)
            if not compare(bins[0], qlines, responses, work, label):
                ok, detail = False, "mismatch (motis compare)"
            elif fell_back is not None:
                ok, detail = False, ("bin%d fell back to CPU on %d/%d queries"
                                     % (fell_back[0], fell_back[1], count))
            else:
                ok, detail = True, "%d ok, gpu_used=%d" % (count, gpu)
            results.append((label, ok, detail))
            fails += 0 if ok else 1
            print("%s %-34s %s" % ("PASS" if ok else "FAIL", label, detail))

    if not results:
        sys.exit("validate: no cases ran -- nothing validated")

    print("%s: %d passed, %d failed" % (a.name, len(results) - fails, fails))

    print_tables(a.name, labels, lat, wall, n_queries)
    sys.exit(1 if fails else 0)


def emit_table(headers, rows, right):
    # aligned (padded) markdown: readable as plain text in a CI log AND valid
    # markdown. `right` = set of column indices to right-align (numbers).
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    def line(cells):
        return "| " + " | ".join(
            (c.rjust(w[i]) if i in right else c.ljust(w[i]))
            for i, c in enumerate(cells)) + " |"
    sep = [("-" * (w[i] - 1) + ":") if i in right else ("-" * w[i])
           for i in range(len(headers))]
    print(line(headers))
    print("| " + " | ".join(sep) + " |")
    for r in rows:
        print(line(r))


def print_tables(name, labels, lat, wall, n_queries):
    # Throughput (aggregate over all cases; motis batch parallelizes queries)
    print("\n### %s throughput (whole batch)\n" % name)
    emit_table(
        ["engine", "queries", "wall_s", "q/s"],
        [[lbl, str(n_queries), "%.1f" % w,
          "%.2f" % (n_queries / w if w > 0 else 0.0)]
         for lbl, w in zip(labels, wall)],
        right={1, 2, 3})

    # Per-query routing-time latency (execute_time, ms) per case + engine
    print("\n### %s per-query latency (execute_time, ms)\n" % name)
    rows = []
    for label, per_bin in lat:
        for lbl, vals in zip(labels, per_bin):
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            rows.append([label, lbl] +
                        ["%.0f" % v for v in (avg, pct(vals, 0.5),
                                              pct(vals, 0.75), pct(vals, 0.9),
                                              pct(vals, 0.99), max(vals))])
    emit_table(["case", "engine", "avg", "q50", "q75", "q90", "q99", "max"],
               rows, right={2, 3, 4, 5, 6, 7})


if __name__ == "__main__":
    main()
