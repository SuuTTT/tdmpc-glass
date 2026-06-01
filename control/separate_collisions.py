#!/usr/bin/env python3
"""
separate_collisions.py — repair CSVs corrupted by duplicate-launch collisions.

Symptom (the "learning curve goes back"): two training processes wrote the SAME
seed CSV concurrently (or one after another), so the `step` column jumps
backward. This de-interleaves the rows into the distinct monotonic runs that
were tangled together.

Method (greedy "closest-behind stream"):
  Walk evaluation rows in (pi, mppi) pairs. Maintain N output streams, each
  required to be strictly increasing in `step`. For each pair at step s, append
  it to the stream whose last step is the *largest value still < s* (the natural
  continuation). If no stream qualifies, open a new stream. This robustly
  separates both the sawtooth-interleave and the sequential-concat patterns
  without assuming a fixed step increment.

Safety contract (per user): NEVER delete or mutate the original.
  - original  seed_N.csv          -> renamed to seed_N.csv.collided  (preserved)
  - run pieces seed_N.run1.csv, seed_N.run2.csv, ...                  (separated)
  - canonical  seed_N.csv          = the most-complete run            (rewritten clean)

Run `--apply` to perform the rename/write; default is a dry run that only reports.
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
from collections import defaultdict


def read_rows(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def step_of(r):
    return int(float(r[0]))


def has_backward_jump(body):
    prev = None
    for r in body:
        try:
            s = step_of(r)
        except Exception:
            continue
        if prev is not None and s < prev - 1:
            return True
        prev = s
    return False


def deinterleave(body):
    """Return a list of streams (each a list of rows), monotonic in step.

    Rows are grouped in consecutive (pi, mppi) eval pairs sharing one step; a
    pair is the atomic unit assigned to a stream so pi/mppi never split apart.
    """
    # group consecutive rows with identical step into one "tick"
    ticks = []
    i = 0
    while i < len(body):
        s = step_of(body[i])
        grp = [body[i]]
        j = i + 1
        while j < len(body) and step_of(body[j]) == s and len(grp) < 2:
            grp.append(body[j])
            j += 1
        ticks.append((s, grp))
        i = j

    streams = []          # list of {"last": int, "rows": [...]}
    for s, grp in ticks:
        # candidate streams whose last step is strictly < s
        cands = [st for st in streams if st["last"] < s]
        if cands:
            tgt = max(cands, key=lambda st: st["last"])  # closest behind
        else:
            tgt = {"last": -1, "rows": []}
            streams.append(tgt)
        tgt["rows"].extend(grp)
        tgt["last"] = s
    return [st["rows"] for st in streams]


def split_by_seed(header, body):
    """For deliberately-interleaved multi-seed files: split on the seed column."""
    try:
        sidx = header.index("seed")
    except ValueError:
        return None
    bins = defaultdict(list)
    for r in body:
        bins[r[sidx]].append(r)
    return dict(bins)


def process(path, apply, by_seed=False, snapshot_ts=None):
    """snapshot_ts set => LIVE-file mode: write de-interleaved side-cars + a
    timestamped backup, but DO NOT rename the original or rewrite the canonical
    (a live writer still holds the inode; the canonical swap is deferred until
    the run completes)."""
    header, body = read_rows(path)
    if not by_seed and not has_backward_jump(body):
        return None
    base = path[:-4] if path.endswith(".csv") else path
    ext = ".csv"

    if by_seed:
        groups = split_by_seed(header, body)
        if not groups:
            print(f"  [skip] {path}: no seed column"); return None
        pieces = {f"{base}.seed{k}{ext}": rows for k, rows in sorted(groups.items())}
        # canonical stays as-is for interleaved files (they are inputs, not seed_N.csv)
        canonical = None
    else:
        streams = deinterleave(body)
        streams.sort(key=lambda rows: -step_of(rows[-1]))  # most-complete first
        pieces = {f"{base}.run{n+1}{ext}": rows for n, rows in enumerate(streams)}
        canonical = streams[0]  # longest-reaching run becomes the clean seed_N.csv

    print(f"  {path}")
    print(f"      original rows={len(body)}  -> {len(pieces)} separated stream(s)")
    for p, rows in pieces.items():
        print(f"        {os.path.basename(p)}: rows={len(rows)} "
              f"steps {step_of(rows[0])}..{step_of(rows[-1])}")
    if canonical is not None:
        print(f"      canonical {os.path.basename(path)} <- run reaching "
              f"{step_of(canonical[-1])} ({len(canonical)} rows); "
              f"original preserved as {os.path.basename(path)}.collided")

    if snapshot_ts:
        print(f"      [LIVE] side-car snapshot only; original left untouched "
              f"(active writer holds inode). backup -> "
              f"{os.path.basename(path)}.collided_snapshot_{snapshot_ts}")

    if not apply:
        return pieces

    if snapshot_ts:
        # LIVE mode: never rename/rewrite the original. Write timestamped backup
        # + de-interleaved side-cars with a distinct suffix so they cannot be
        # confused with the canonical or clobber a future post-completion fix.
        import shutil
        shutil.copy2(path, f"{path}.collided_snapshot_{snapshot_ts}")
        streams = deinterleave(body)
        streams.sort(key=lambda rows: -step_of(rows[-1]))
        for n, rows in enumerate(streams):
            p = f"{base}.snapshot_{snapshot_ts}.run{n+1}{ext}"
            with open(p, "w", newline="") as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)
        return pieces

    # write separated pieces
    for p, rows in pieces.items():
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    if canonical is not None:
        collided = path + ".collided"
        if not os.path.exists(collided):
            os.rename(path, collided)            # preserve original (renamed, not deleted)
        with open(path, "w", newline="") as f:    # rewrite canonical clean
            w = csv.writer(f); w.writerow(header); w.writerows(canonical)
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="exp/tdmpc_glass")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--snapshot-live", nargs="+", metavar="CSV",
                    help="LIVE files: write de-interleaved side-car snapshots + a "
                         "timestamped backup WITHOUT renaming/rewriting the original "
                         "(use while a writer still holds the file).")
    ap.add_argument("--ts", help="timestamp tag for --snapshot-live backups")
    args = ap.parse_args()

    if args.snapshot_live:
        ts = args.ts or "snapshot"
        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"=== [{mode}] LIVE snapshot (non-destructive) ts={ts} ===\n")
        for f in args.snapshot_live:
            process(f, args.apply, by_seed=False, snapshot_ts=ts)
        print(f"\n{mode} complete.")
        return

    # files where the seed column is the intended separator (deliberate combos)
    by_seed_markers = ("interleaved", "rerun")

    collisions, interleaved = [], []
    for f in glob.glob(args.root + "/**/*.csv", recursive=True):
        try:
            header, body = read_rows(f)
        except Exception:
            continue
        if header and header[0] == "series":      # plot files: legit multi-series
            continue
        if len(body) < 3:
            continue
        if not has_backward_jump(body):
            continue
        if any(m in f for m in by_seed_markers):
            interleaved.append(f)
        else:
            collisions.append(f)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== [{mode}] collision repair under {args.root} ===\n")
    print(f"per-seed collisions ({len(collisions)}):")
    for f in sorted(collisions):
        process(f, args.apply, by_seed=False)
    print(f"\ndeliberate interleaved/rerun files ({len(interleaved)}):")
    for f in sorted(interleaved):
        process(f, args.apply, by_seed=True)
    print(f"\n{mode} complete.")


if __name__ == "__main__":
    main()
