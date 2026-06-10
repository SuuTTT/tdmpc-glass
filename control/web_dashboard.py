"""TD-MPC-Glass live web dashboard — thin entrypoint.

The dashboard was refactored from a single 2887-line monolith into the
``control/dashboard/`` package:

  dashboard/__init__.py        — Flask app factory + /api/boxes /api/curves /api/phases
  dashboard/boxprobe.py        — fleet registry (imported from the daemon) + SSH probing
  dashboard/data.py            — CSV discovery, eval summaries, phase/CI stats
  dashboard/queue_api.py       — /api/queue blueprint
  dashboard/templates/index.html — the dashboard page

This file is kept as the launch target so ``control/start_center.sh`` and
``python control/web_dashboard.py`` (with the DASHBOARD_PORT env var, default
5055) keep working unchanged.

Run:
  /home/ubuntu/tdmpc-glass/.venv/bin/python control/web_dashboard.py
"""
from __future__ import annotations

import os
import sys

# The package uses flat imports (boxprobe, data, queue_api), so put both the
# package dir and the control dir (for the read-only daemon BOXES import) on path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.join(_THIS_DIR, "dashboard")
for _p in (_PKG_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard import create_app, start_box_refresher, BOX_REFRESH_S  # noqa: E402

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("DASHBOARD_PORT", 5055))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    # Kick off the background box-probe refresher so /api/boxes serves from cache.
    start_box_refresher()
    print(f"[web_dashboard] serving on http://{host}:{port} (box cache refresh {BOX_REFRESH_S}s)")
    app.run(host=host, port=port, debug=False, threaded=True)
