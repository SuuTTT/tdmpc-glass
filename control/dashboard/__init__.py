"""TD-MPC-Glass live web dashboard (Flask app factory + core routes).

Serves a single HTML page showing per-box fleet status (SSH-probed), learning
curves of every active *_phase*/seed_*.csv, and the central task queue.

Package layout:
  __init__.py   — Flask app + page route + /api/boxes /api/curves /api/phases
  boxprobe.py   — fleet registry (imported from the daemon) + SSH probing
  data.py       — CSV discovery, eval summaries, phase/CI stats
  queue_api.py  — /api/queue blueprint
  templates/index.html — the dashboard page (Jinja-rendered)

The fleet box list is the single source of truth from
control/task_queue_daemon.py; see boxprobe.BOXES.
"""
from __future__ import annotations

import json
import os
import threading
import time

from flask import Flask, jsonify, render_template, request

from boxprobe import (
    BOXES,
    probe_box,
    parse_etime_seconds,
    _BOX_CACHE,
    _BOX_CACHE_LOCK,
)
from data import (
    discover_csvs,
    eval_summary,
    fmt_metric,
    read_diag_last_mppi,
    build_curves,
    build_phases,
    build_phase_ci,
)
from queue_api import queue_bp


# ── Background box-probe cache ──────────────────────────────────────────────
# /api/boxes used to SSH-probe all boxes synchronously (~15s/request). A daemon
# thread refreshes the snapshot every BOX_REFRESH_S so the route returns from cache.
_BOXES_SNAPSHOT = {"payload": None, "ts": 0.0}
_BOXES_SNAP_LOCK = threading.Lock()
BOX_REFRESH_S = int(os.environ.get("BOX_REFRESH_S", "20"))

# Hand-written "what's live right now" blurbs (experiment + dev), maintained by
# the monitor loop. Missing/corrupt file → the panel shows "status unavailable".
LIVE_STATUS_FILE = "/home/ubuntu/tdmpc-glass/exp/tdmpc_glass/live_status.json"


def _build_boxes_payload():
    boxes = []
    threads, results = [], {}

    def worker(entry):
        tag, port, host, gpu_idx, label = entry
        results[tag] = probe_box(tag, port, host, gpu_idx)

    for entry in BOXES:
        t = threading.Thread(target=worker, args=(entry,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=20)
    # Compute CSV index once so we annotate every proc against the same snapshot.
    csv_index = discover_csvs()
    by_phase_seed = {(c["phase"], c["seed"]): c for c in csv_index}
    by_box_phase_seed = {(c["box"], c["phase"], c["seed"]): c for c in csv_index}
    for tag, port, host, gpu_idx, label in BOXES:
        info = results.get(tag, {"reachable": False, "error": "no-result", "procs": []})
        # If this is a slot on a multi-GPU box, drop procs not pinned to this CUDA index.
        if any(p.get("cuda_visible") for p in info.get("procs", [])):
            info["procs"] = [p for p in info["procs"]
                             if p.get("cuda_visible", "") == str(gpu_idx)
                             or not p.get("cuda_visible")]
        # Dedupe: if two procs share (seed, output_tag), keep the longer-running one.
        deduped = {}
        for p in info.get("procs", []):
            key = (p.get("seed"), p.get("output_tag"))
            prev = deduped.get(key)
            if prev is None or len(p.get("etime", "")) > len(prev.get("etime", "")):
                if prev is not None:
                    p["dup_count"] = prev.get("dup_count", 1) + 1
                else:
                    p["dup_count"] = 1
                deduped[key] = p
            else:
                prev["dup_count"] = prev.get("dup_count", 1) + 1
        info["procs"] = list(deduped.values())
        for p in info.get("procs", []):
            seed = p.get("seed", "?")
            phase_from_env = p.get("output_tag") or ""
            picked = None
            if phase_from_env:
                picked = (by_box_phase_seed.get((tag, phase_from_env, seed))
                          or by_phase_seed.get((phase_from_env, seed)))
                if picked is None:
                    p["phase"] = phase_from_env
                    p["best_pi"] = p["best_mppi"] = p["best_any"] = None
                    p["last_pi"] = p["last_mppi"] = None
                    p["best_pi_step"] = p["best_mppi_step"] = p["best_any_step"] = None
                    p["last_pi_step"] = p["last_mppi_step"] = None
                    p["pi_minus_mppi_last"] = None
                    continue
            if picked is None:
                same_seed = [c for c in csv_index if c["seed"] == seed]
                on_box = [c for c in same_seed if c["box"] == tag]
                cand = on_box or same_seed
                cand.sort(key=lambda r: r["mtime"], reverse=True)
                picked = cand[0] if cand else None
            if picked:
                summary = eval_summary(picked["path"])
                p["phase"] = picked["phase"]
                p["best_pi"] = fmt_metric(summary["best_pi"])
                p["best_pi_step"] = summary["best_pi_step"] if summary["best_pi_step"] >= 0 else None
                p["best_mppi"] = fmt_metric(summary["best_mppi"])
                p["best_mppi_step"] = summary["best_mppi_step"] if summary["best_mppi_step"] >= 0 else None
                p["best_any"] = fmt_metric(summary["best_any"])
                p["best_any_step"] = summary["best_any_step"] if summary["best_any_step"] >= 0 else None
                p["best_any_selector"] = summary["best_any_selector"]
                p["last_pi"] = fmt_metric(summary["last_pi"])
                p["last_pi_step"] = summary["last_pi_step"] if summary["last_pi_step"] >= 0 else None
                p["last_mppi"] = fmt_metric(summary["last_mppi"])
                p["last_mppi_step"] = summary["last_mppi_step"] if summary["last_mppi_step"] >= 0 else None
                p["last_step"] = p["last_mppi_step"] or p["last_pi_step"]
                p["pi_minus_mppi_last"] = (
                    round(summary["pi_minus_mppi_last"], 1)
                    if summary["pi_minus_mppi_last"] is not None else None
                )
                diag_path = picked["path"].replace(".csv", "_diag.csv")
                p["diag"] = read_diag_last_mppi(diag_path)
            else:
                p["phase"] = None
                p["best_pi"] = p["best_mppi"] = p["best_any"] = None
                p["last_pi"] = p["last_mppi"] = None
                p["best_pi_step"] = p["best_mppi_step"] = p["best_any_step"] = None
                p["last_pi_step"] = p["last_mppi_step"] = p["last_step"] = None
                p["pi_minus_mppi_last"] = None
                p["diag"] = None
            # Approximate live SPS = last_step / (etime - JIT_warmup).
            et = parse_etime_seconds(p.get("etime", ""))
            last_step = p.get("last_step")
            if et and et > 60 and last_step and last_step > 0:
                JIT_WARMUP_S = 60
                effective = max(et - JIT_WARMUP_S, 1)
                p["sps_avg"] = int(last_step / effective)
            else:
                p["sps_avg"] = None
        boxes.append({"tag": tag, "label": label, "host": host, "port": port,
                      "gpu_idx": gpu_idx, **info})
    active = sorted({(p["phase"], p["seed"]) for b in boxes for p in b.get("procs", [])
                    if p.get("phase") and p.get("seed")})
    with _BOX_CACHE_LOCK:
        for b in boxes:
            _BOX_CACHE[b["tag"]] = b
    return {"boxes": boxes, "active": [{"phase": p, "seed": s} for p, s in active],
            "ts": time.time()}


def _store_boxes_snapshot(payload):
    with _BOXES_SNAP_LOCK:
        _BOXES_SNAPSHOT["payload"] = payload
        _BOXES_SNAPSHOT["ts"] = time.time()


def _boxes_refresher_loop():
    while True:
        try:
            _store_boxes_snapshot(_build_boxes_payload())
        except Exception as e:
            print(f"[web_dashboard] box refresh error: {e}", flush=True)
        time.sleep(BOX_REFRESH_S)


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(queue_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/boxes")
    def api_boxes():
        with _BOXES_SNAP_LOCK:
            payload = _BOXES_SNAPSHOT["payload"]
            ts = _BOXES_SNAPSHOT["ts"]
        if payload is None:
            # First request before the refresher produced a snapshot: build once.
            payload = _build_boxes_payload()
            _store_boxes_snapshot(payload)
            ts = _BOXES_SNAPSHOT["ts"]
        out = dict(payload)
        out["cache_age_s"] = round(time.time() - ts, 1)
        return jsonify(out)

    @app.route("/api/live_status")
    def api_live_status():
        try:
            with open(LIVE_STATUS_FILE) as f:
                d = json.load(f)
            return jsonify({
                "ok": True,
                "updated": d.get("updated"),
                "experiment": d.get("experiment"),
                "dev": d.get("dev"),
                "eta": d.get("eta"),
                "history": d.get("history", []),
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})

    @app.route("/api/curves")
    def api_curves():
        out = build_curves(request.args.get("phase"))
        return jsonify({"curves": out, "ts": time.time()})

    @app.route("/api/phases")
    def api_phases():
        return jsonify({"phases": build_phases()})

    @app.route("/api/phase_ci")
    def api_phase_ci():
        return jsonify({"ci_curves": build_phase_ci(request.args.get("phase", ""))})

    return app


def start_box_refresher():
    """Launch the background box-probe refresher thread (call once at startup)."""
    threading.Thread(target=_boxes_refresher_loop, daemon=True).start()
