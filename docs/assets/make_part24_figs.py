#!/usr/bin/env python3
"""Emit the Part 24 figures as inline-able SVG. Stdlib only, theme-aware.

Same palette contract as make_part23_figs.py:
  slot1 blue #2a78d6 = planning ON throughout
  slot2 orange #eb6834 = planning OFF throughout
  slot3 aqua  #1baf7a = the gate (planning switched on partway)
Dark-mode steps come from the same ramps via a <style> block, so figures follow
the reader's theme rather than being flipped.
"""
import os

STYLE = """<style>
  .surf{fill:#fcfcfb}
  .ink{fill:#0b0b0b}.ink2{fill:#52514e}
  .grid{stroke:#dcdbd7;stroke-width:1}
  .axis{stroke:#a9a8a3;stroke-width:1}
  .s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.s4{fill:#eda100}
  .l1{stroke:#2a78d6}.l2{stroke:#eb6834}.l3{stroke:#1baf7a}
  .dash{stroke:#a9a8a3;stroke-width:1.5;stroke-dasharray:5 4}
  .dot1{fill:#2a78d6}.dot2{fill:#eb6834}.dot3{fill:#1baf7a}
  .ln{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
  text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
  @media (prefers-color-scheme:dark){
    .surf{fill:#1a1a19}.ink{fill:#ffffff}.ink2{fill:#c3c2b7}
    .grid{stroke:#3a3a38}.axis{stroke:#5c5b57}
    .s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}.s4{fill:#c98500}
    .l1{stroke:#3987e5}.l2{stroke:#d95926}.l3{stroke:#199e70}
    .dot1{fill:#3987e5}.dot2{fill:#d95926}.dot3{fill:#199e70}
    .dash{stroke:#5c5b57}
  }
</style>"""


def hdr(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" role="img" aria-label="{title}">{STYLE}'
            f'<rect class="surf" width="{w}" height="{h}" rx="6"/>')


# per-seed values, 400k steps, official TD-MPC2
TASKS = [
    ("cup-catch",   [982.80],                                 [981.50],                  True),
    ("finger-spin", [985.00],                                  [984.70],                 True),
    ("cheetah-run", [910.90, 640.94, 915.61, 903.59, 908.42],  [910.40, 907.66, 901.19, 808.70], False),
    ("walker-run",  [822.27, 808.41, 799.06],                  [699.19, 750.16, 622.52], False),
    ("acrobot-sw.", [504.90],                                  [409.20],                 False),
    ("hopper-hop",  [499.39, 349.58, 333.43],                  [45.57, 119.98, 42.83],   False),
]


def fig1(path):
    """Per-task: planning ON vs OFF, bars = mean, dots = individual seeds."""
    W, H = 760, 420
    x0, y0, pw, ph = 66, 96, 660, 250
    ymax = 1000
    s = [hdr(W, H, "Effect of removing the planner, six DMControl tasks")]
    s.append('<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             'Removing MPPI planning: free on some tasks, fatal on others</text>')
    s.append('<text class="ink2" x="16" y="45" font-size="12">'
             'Official TD-MPC2, 400k env steps. Bars = mean, dots = individual seeds.</text>')
    s.append('<text class="ink2" x="16" y="62" font-size="12">'
             'Grey hatched labels mark tasks at their score ceiling, where the comparison carries no information.</text>')
    for i in range(6):
        v = i * 200
        y = y0 + ph - ph * v / ymax
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    gw = pw / len(TASKS)
    for gi, (task, plan, never, ceiling) in enumerate(TASKS):
        cx = x0 + gi * gw + gw / 2
        for bi, (vals, cls, dcls) in enumerate(((plan, "s1", "dot1"), (never, "s2", "dot2"))):
            m = sum(vals) / len(vals)
            bw = 30
            bx = cx - bw - 4 + bi * (bw + 8)
            bh = ph * m / ymax
            by = y0 + ph - bh
            s.append(f'<rect class="{cls}" x="{bx:.1f}" y="{by:.1f}" width="{bw}" '
                     f'height="{bh:.1f}" rx="3" opacity="0.85"/>')
            s.append(f'<text class="ink" x="{bx+bw/2:.1f}" y="{by-6:.1f}" font-size="10.5" '
                     f'font-weight="600" text-anchor="middle">{m:.0f}</text>')
            for v in vals:
                cy = y0 + ph - ph * v / ymax
                s.append(f'<circle class="{dcls}" cx="{bx+bw/2:.1f}" cy="{cy:.1f}" r="2.6" '
                         f'stroke="#fcfcfb" stroke-width="0.8"/>')
        lab_cls = "ink2" if ceiling else "ink"
        s.append(f'<text class="{lab_cls}" x="{cx:.1f}" y="{y0+ph+20:.0f}" font-size="11" '
                 f'font-weight="{"400" if ceiling else "600"}" text-anchor="middle">{task}</text>')
        if ceiling:
            s.append(f'<text class="ink2" x="{cx:.1f}" y="{y0+ph+34:.0f}" font-size="9.5" '
                     f'text-anchor="middle" font-style="italic">ceiling</text>')
        else:
            n = f"n={len(plan)}/{len(never)}"
            s.append(f'<text class="ink2" x="{cx:.1f}" y="{y0+ph+34:.0f}" font-size="9.5" '
                     f'text-anchor="middle">{n}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    ly = H - 14
    s.append(f'<rect class="s1" x="{x0}" y="{ly-9}" width="11" height="11" rx="2"/>')
    s.append(f'<text class="ink2" x="{x0+17}" y="{ly}" font-size="11.5">planning ON throughout (stock TD-MPC2)</text>')
    s.append(f'<rect class="s2" x="{x0+300}" y="{ly-9}" width="11" height="11" rx="2"/>')
    s.append(f'<text class="ink2" x="{x0+317}" y="{ly}" font-size="11.5">planning OFF throughout</text>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


HOP = {
    "plan":  [(50000, 0.0), (100000, 59.5), (150000, 101.5), (200000, 258.1), (250000, 315.5), (400000, 394.1)],
    "gate":  [(50000, 7.5), (100000, 93.3), (150000, 38.4), (175000, 183.1), (200000, 115.9), (300000, 298.7), (400000, 250.4)],
    "never": [(50000, 7.5), (100000, 59.5), (150000, 87.4), (200000, 95.2), (250000, 86.6), (400000, 69.5)],
}


def fig2(path):
    """hopper-hop: the three arms over training, and where the gate fires."""
    W, H = 760, 400
    x0, y0, pw, ph = 66, 88, 620, 250
    xmax, ymax = 400000, 420
    s = [hdr(W, H, "hopper-hop: planning, no planning, and the gate")]
    s.append('<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             'hopper-hop: the planner buys nothing until ~150k, then it unlocks the task</text>')
    s.append('<text class="ink2" x="16" y="45" font-size="12">'
             'Mean return across seeds at each evaluation. The gate arm acts from the policy prior until 150k, then plans.</text>')
    s.append('<text class="ink2" x="16" y="62" font-size="12">'
             'Below ~150k the planning and no-planning arms are indistinguishable — the model is not yet good enough to plan through.</text>')
    for i in range(5):
        v = i * 100
        y = y0 + ph - ph * v / ymax
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    for st in (0, 100000, 200000, 300000, 400000):
        x = x0 + pw * st / xmax
        s.append(f'<text class="ink2" x="{x:.1f}" y="{y0+ph+18:.0f}" font-size="10.5" '
                 f'text-anchor="middle">{st//1000}k</text>')
    gx = x0 + pw * 150000 / xmax
    s.append(f'<line class="dash" x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0+ph}"/>')
    s.append(f'<text class="ink2" x="{gx+6:.1f}" y="{y0+13:.0f}" font-size="10.5">gate fires (150k)</text>')
    for key, cls in (("plan", "l1"), ("never", "l2"), ("gate", "l3")):
        pts = " ".join(f"{x0+pw*st/xmax:.1f},{y0+ph-ph*min(v,ymax)/ymax:.1f}" for st, v in HOP[key])
        s.append(f'<polyline class="ln {cls}" points="{pts}"/>')
    for key, cls, lab in (("plan", "ink", "planning throughout — 394"),
                          ("gate", "ink", "gate at 150k — 250"),
                          ("never", "ink", "never planning — 70")):
        st, v = HOP[key][-1]
        x = x0 + pw * st / xmax
        y = y0 + ph - ph * min(v, ymax) / ymax
        s.append(f'<text class="{cls}" x="{x+8:.1f}" y="{y+4:.1f}" font-size="11" font-weight="600">{lab}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    s.append(f'<text class="ink2" x="{x0+pw/2:.0f}" y="{y0+ph+38:.0f}" font-size="11.5" '
             f'text-anchor="middle">environment steps</text>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


def fig3(path):
    """The Pareto view: return against wall-clock for the three hopper arms."""
    W, H = 720, 380
    x0, y0, pw, ph = 78, 84, 560, 230
    s = [hdr(W, H, "Return against wall-clock, hopper-hop")]
    s.append('<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             'The gate is a Pareto point, not a free lunch</text>')
    s.append('<text class="ink2" x="16" y="45" font-size="12">'
             'hopper-hop, same box (wall-clock is host-sensitive, so all three are from box1).</text>')
    s.append('<text class="ink2" x="16" y="62" font-size="12">'
             'Up and to the left is better: more return, less time.</text>')
    pts = [("planning throughout", 167, 394.1, "s1", "dot1"),
           ("gate at 150k", 146, 250.4, "s3", "dot3"),
           ("never planning", 109, 69.5, "s2", "dot2")]
    tmin, tmax, rmax = 90, 185, 420
    for i in range(5):
        v = i * 100
        y = y0 + ph - ph * v / rmax
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    for t in (100, 120, 140, 160, 180):
        x = x0 + pw * (t - tmin) / (tmax - tmin)
        s.append(f'<text class="ink2" x="{x:.1f}" y="{y0+ph+18:.0f}" font-size="10.5" '
                 f'text-anchor="middle">{t}m</text>')
    poly = " ".join(f"{x0+pw*(t-tmin)/(tmax-tmin):.1f},{y0+ph-ph*r/rmax:.1f}" for _, t, r, _, _ in pts)
    s.append(f'<polyline class="ln" points="{poly}" stroke="#a9a8a3" stroke-width="1.5" stroke-dasharray="4 4"/>')
    for lab, t, r, cls, dcls in pts:
        x = x0 + pw * (t - tmin) / (tmax - tmin)
        y = y0 + ph - ph * r / rmax
        s.append(f'<circle class="{dcls}" cx="{x:.1f}" cy="{y:.1f}" r="7"/>')
        anchor = "end" if lab == "planning throughout" else "start"
        dx = -12 if anchor == "end" else 12
        s.append(f'<text class="ink" x="{x+dx:.1f}" y="{y+4:.1f}" font-size="11.5" '
                 f'font-weight="600" text-anchor="{anchor}">{lab}</text>')
        s.append(f'<text class="ink2" x="{x+dx:.1f}" y="{y+19:.1f}" font-size="10.5" '
                 f'text-anchor="{anchor}">{r:.0f} return · {t}min</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    s.append(f'<text class="ink2" x="{x0+pw/2:.0f}" y="{y0+ph+38:.0f}" font-size="11.5" '
             f'text-anchor="middle">wall-clock for 400k steps (minutes)</text>')
    s.append(f'<text class="ink2" x="18" y="{y0+ph/2:.0f}" font-size="11.5" '
             f'transform="rotate(-90 18 {y0+ph/2:.0f})" text-anchor="middle">return at 400k</text>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    fig1(os.path.join(d, "part24-per-task.svg"))
    fig2(os.path.join(d, "part24-hopper-curves.svg"))
    fig3(os.path.join(d, "part24-pareto.svg"))
    print("wrote part24-per-task.svg, part24-hopper-curves.svg, part24-pareto.svg")
