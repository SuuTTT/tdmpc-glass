#!/usr/bin/env python3
"""Regression asserts for WON phases. An edit may be kept only if these still
pass (won phases must not regress). Reads a metrics JSON produced by harness.py.

Usage: python regression.py m_vN.json
"""
import sys, json

# Won-phase floors. Update (loosen never; tighten as phases are conquered).
FLOORS = {
    "approach_ok": 0.95,
    "descend_ok": 0.95,
    "grasp_ok": 0.90,
    "reached_box_rate": 0.70,
    "lift_ok": 0.45,
    "success_rate": 0.03,   # we have crossed 0% — never regress below a real pick
}


def check(metrics):
    fails = []
    for k, floor in FLOORS.items():
        v = metrics.get(k, 0.0)
        if v < floor:
            fails.append(f"REGRESSION {k}={v} < floor {floor}")
    return fails


if __name__ == "__main__":
    m = json.loads(open(sys.argv[1]).read())
    fails = check(m)
    if fails:
        print("\n".join(fails)); sys.exit(1)
    print("regression OK:", {k: m.get(k) for k in FLOORS}); sys.exit(0)
