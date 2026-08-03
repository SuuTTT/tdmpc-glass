#!/usr/bin/env python3
"""Emit the Part 23 figures as inline-able SVG. Stdlib only.

Palette = the validated categorical slots, assigned in fixed order:
  slot1 blue #2a78d6, slot2 orange #eb6834, slot3 aqua #1baf7a, slot4 yellow #eda100
Dark-mode steps come from the same ramps and are applied via a <style> block inside
each SVG, so the figures follow the reader's theme instead of being flipped.
"""
import os

STYLE = """<style>
  .surf{fill:#fcfcfb}
  .ink{fill:#0b0b0b}.ink2{fill:#52514e}
  .grid{stroke:#dcdbd7;stroke-width:1}
  .axis{stroke:#a9a8a3;stroke-width:1}
  .s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.s4{fill:#eda100}
  .l1{stroke:#2a78d6}.l2{stroke:#eb6834}.l3{stroke:#1baf7a}
  .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
  text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
  @media (prefers-color-scheme:dark){
    .surf{fill:#1a1a19}.ink{fill:#ffffff}.ink2{fill:#c3c2b7}
    .grid{stroke:#3a3a38}.axis{stroke:#5c5b57}
    .s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}.s4{fill:#c98500}
    .l1{stroke:#3987e5}.l2{stroke:#d95926}.l3{stroke:#199e70}
  }
</style>"""


def hdr(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" role="img" aria-label="{title}">{STYLE}'
            f'<rect class="surf" width="{w}" height="{h}" rx="6"/>')


def fig1(path):
    """Regime: the same ablation under two data-collection conditions."""
    W, H = 720, 330
    x0, y0, pw, ph = 78, 60, 600, 190
    ymax = 1000
    s = []
    s.append(hdr(W, H, "World-model ablation under two data regimes"))
    s.append(f'<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             f'Same ablation, same code — only the data regime differs</text>')
    s.append(f'<text class="ink2" x="16" y="45" font-size="12">'
             f'cheetah-run, official TD-MPC2, n=3 each. Bars are mean return.</text>')
    for i in range(6):
        v = i * 200
        y = y0 + ph - ph * v / ymax
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{v}</text>')
    groups = [("on-policy\n(each arm collects its own data)", 907.4, 869.2, "1.04x"),
              ("matched-data\n(one shared frozen policy)", 615.0, 173.0, "3.55x")]
    gw = pw / 2
    for gi, (lab, full, strip, ratio) in enumerate(groups):
        cx = x0 + gi * gw + gw / 2
        for bi, (val, cls) in enumerate(((full, "s1"), (strip, "s2"))):
            bw = 78
            bx = cx - bw - 8 + bi * (bw + 16)
            bh = ph * val / ymax
            by = y0 + ph - bh
            s.append(f'<rect class="{cls}" x="{bx:.1f}" y="{by:.1f}" width="{bw}" '
                     f'height="{bh:.1f}" rx="4"/>')
            s.append(f'<text class="ink" x="{bx+bw/2:.1f}" y="{by-7:.1f}" font-size="12" '
                     f'font-weight="600" text-anchor="middle">{val:.0f}</text>')
        s.append(f'<text class="ink" x="{cx:.1f}" y="{y0+ph+22:.0f}" font-size="12" '
                 f'font-weight="600" text-anchor="middle">{lab.splitlines()[0]}</text>')
        s.append(f'<text class="ink2" x="{cx:.1f}" y="{y0+ph+38:.0f}" font-size="10.5" '
                 f'text-anchor="middle">{lab.splitlines()[1]}</text>')
        s.append(f'<text class="ink" x="{cx:.1f}" y="{y0-18:.0f}" font-size="14" '
                 f'font-weight="700" text-anchor="middle">{ratio}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    ly = H - 16
    s.append(f'<rect class="s1" x="{x0}" y="{ly-9}" width="11" height="11" rx="2"/>')
    s.append(f'<text class="ink2" x="{x0+17}" y="{ly}" font-size="11.5">world model ON (cc=20)</text>')
    s.append(f'<rect class="s2" x="{x0+185}" y="{ly-9}" width="11" height="11" rx="2"/>')
    s.append(f'<text class="ink2" x="{x0+202}" y="{ly}" font-size="11.5">world model OFF (cc=0)</text>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


def fig2(path):
    """Prunability: three tasks, three curve shapes."""
    W, H = 720, 360
    x0, y0, pw, ph = 62, 62, 600, 220
    s = [hdr(W, H, "Dose-response of the world-model loss weight, three tasks")]
    s.append(f'<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             f'How much of the objective can you remove? Three tasks, three shapes</text>')
    s.append(f'<text class="ink2" x="16" y="45" font-size="12">'
             f'DreamerV3, return as % of that task’s own full-objective score.</text>')
    xs = [1.0, 0.5, 0.25, 0.1, 0.0]
    series = [("reacher — cliff", [100, 100.0, 99.4, 12.1, 4.7], "l1", "s1"),
              ("acrobot — ramp", [100, 88.3, 69.9, 36.3, 24.8], "l2", "s2"),
              ("cheetah — basin", [100, 101.9, 89.0, 91.7, 82.7], "l3", "s3")]
    def px(i):  return x0 + pw * i / (len(xs) - 1)
    def py(v):  return y0 + ph - ph * min(v, 110) / 110
    for g in range(0, 111, 25):
        y = py(g)
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{g}%</text>')
    for i, xv in enumerate(xs):
        s.append(f'<text class="ink2" x="{px(i):.1f}" y="{y0+ph+20:.0f}" font-size="11.5" '
                 f'text-anchor="middle">{xv:g}</text>')
    s.append(f'<text class="ink2" x="{x0+pw/2:.0f}" y="{y0+ph+40:.0f}" font-size="12" '
             f'text-anchor="middle">world-model loss weight (dyn scale) — 1.0 is the default, 0 removes it</text>')
    for name, ys, lc, fc in series:
        d = " ".join(("M" if i == 0 else "L") + f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
        s.append(f'<path class="ln {lc}" d="{d}"/>')
        for i, v in enumerate(ys):
            s.append(f'<circle class="{fc}" cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5"/>')
    ly = H - 16
    for i, (name, _, _, fc) in enumerate(series):
        lx = x0 + i * 200
        s.append(f'<rect class="{fc}" x="{lx}" y="{ly-9}" width="11" height="11" rx="2"/>')
        s.append(f'<text class="ink2" x="{lx+17}" y="{ly}" font-size="11.5">{name}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


def fig3(path):
    """Efficiency: return vs throughput, four configs."""
    W, H = 720, 330
    x0, y0, pw, ph = 72, 58, 590, 200
    s = [hdr(W, H, "Return versus throughput for four configurations")]
    s.append(f'<text class="ink" x="16" y="26" font-size="15" font-weight="600">'
             f'Two config lines: 1.9× the speed, no loss in return</text>')
    s.append(f'<text class="ink2" x="16" y="45" font-size="12">'
             f'cheetah-run, 400k steps. Up and to the right is better.</text>')
    pts = [("default", 35.8, 862.4, "s2"), ("depth 2–3", 50.0, 910.6, "s4"),
           ("update ratio 2", 49.0, 892.3, "s3"), ("both", 67.9, 908.9, "s1")]
    xlo, xhi, ylo, yhi = 30, 75, 830, 940
    def px(v): return x0 + pw * (v - xlo) / (xhi - xlo)
    def py(v): return y0 + ph - ph * (v - ylo) / (yhi - ylo)
    for g in range(840, 941, 25):
        y = py(g)
        s.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}"/>')
        s.append(f'<text class="ink2" x="{x0-8}" y="{y+4:.1f}" font-size="11" text-anchor="end">{g}</text>')
    for g in (30, 40, 50, 60, 70):
        x = px(g)
        s.append(f'<text class="ink2" x="{x:.1f}" y="{y0+ph+20:.0f}" font-size="11" text-anchor="middle">{g}</text>')
    s.append(f'<text class="ink2" x="{x0+pw/2:.0f}" y="{y0+ph+40:.0f}" font-size="12" '
             f'text-anchor="middle">throughput (env steps / second)</text>')
    s.append(f'<text class="ink2" x="16" y="{y0+ph/2:.0f}" font-size="12" '
             f'transform="rotate(-90 16 {y0+ph/2:.0f})" text-anchor="middle">return @400k</text>')
    for name, xv, yv, fc in pts:
        s.append(f'<circle class="{fc}" cx="{px(xv):.1f}" cy="{py(yv):.1f}" r="9"/>')
        s.append(f'<text class="ink" x="{px(xv):.1f}" y="{py(yv)-16:.1f}" font-size="12" '
                 f'font-weight="600" text-anchor="middle">{name}</text>')
    s.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    s.append("</svg>")
    open(path, "w").write("".join(s))


here = os.path.dirname(os.path.abspath(__file__))
fig1(os.path.join(here, "part23-regime.svg"))
fig2(os.path.join(here, "part23-prunability.svg"))
fig3(os.path.join(here, "part23-efficiency.svg"))
print("wrote 3 svg figures")
