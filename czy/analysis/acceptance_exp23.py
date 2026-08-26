"""exp2.3 acceptance analysis: swing clearance closed-loop verification.

Usage: python acceptance_exp23.py <csv_path>
Metrics per segment/foot:
  - steps, cycle vs theory (freq ratio <= 1.5x required)
  - lift P50/P90 (P50 >= 2.5cm required)
  - mean swing gap (>= 2cm required)
  - t(z>3cm) fraction of swing time (>= 15% required)
  - mid-swing touches (0 required)
Global: fall count (base_height < 0.45m).
"""
import csv
import sys
import statistics as st

PATH = sys.argv[1]
N_SEG = 6
CYC_MIN = 0.35
CYC_SLOPE = (0.7 - 0.35) / 0.6  # 0.5833 s per m/s
THRESH = 1.0
MIN_SWING = 5
DT = 0.01
PEAK_REQ = 0.03   # 3cm time-above threshold
LIFT_P50_REQ = 0.025
GAP_REQ = 0.02

rows = list(csv.DictReader(open(PATH)))
N = len(rows)
fl = [float(r["foot_z_l"]) for r in rows]
fr = [float(r["foot_z_r"]) for r in rows]
pl = [float(r["foot_force_l"]) for r in rows]
pr = [float(r["foot_force_r"]) for r in rows]
v = [float(r["base_vel_x"]) for r in rows]
h = [float(r["base_height"]) for r in rows]

cbL = sum(z for z, f in zip(fl, pl) if f > 50) / max(1, sum(1 for f in pl if f > 50))
cbR = sum(z for z, f, in zip(fr, pr) if f > 50) / max(1, sum(1 for f in pr if f > 50))
base = (cbL + cbR) / 2
falls = sum(1 for i in range(1, N) if h[i] < 0.45 and h[i] <= h[i - 1])
print(f"rows={N} contact_baseline={base*100:.1f}cm fall_frames={falls} h_min={min(h):.3f}m")
print(f"acceptance: P50>={LIFT_P50_REQ*100:.0f}cm  gap>={GAP_REQ*100:.0f}cm  t(>3cm)>={15}%  freq<=1.5x  touches=0")


def swings_of(f, i0, i1):
    sw = []
    in_sw = False
    s0 = 0
    for i in range(i0, i1 + 1):
        c = f[i] > THRESH
        if not c:
            if not in_sw:
                in_sw = True
                s0 = i
        else:
            if in_sw:
                in_sw = False
                if i - s0 >= MIN_SWING:
                    sw.append((s0, i))
    if in_sw and i1 - s0 >= MIN_SWING:
        sw.append((s0, i1))
    return sw


def analyze(z, f, side, i0, i1, dur_s):
    sw = swings_of(f, i0, i1)
    n_sw = len(sw)
    lifts = [max(z[a:b]) - base for (a, b) in sw]
    # mean swing gap: mean clearance over whole swing duration
    gaps = [st.mean(z[a:b]) - base for (a, b) in sw]
    # time above 3cm within swing
    t_frac = []
    for (a, b) in sw:
        above = sum(1 for i in range(a, b) if z[i] - base > PEAK_REQ)
        t_frac.append(above / max(1, b - a))
    # mid-swing touches (sandwiched force bursts)
    n_touch = 0
    for (a, b) in sw:
        i = a
        while i < b:
            if f[i] > THRESH:
                j = i
                while j < b and f[j] > THRESH:
                    j += 1
                if j < b:
                    n_touch += 1
                i = j
            else:
                i += 1
    period = dur_s / n_sw if n_sw else 0
    p50 = st.median(lifts) * 100 if lifts else 0
    p90 = (sorted(lifts)[int(0.9 * len(lifts))] * 100) if lifts else 0
    mg = st.mean(gaps) * 100 if gaps else 0
    tf = st.mean(t_frac) * 100 if t_frac else 0
    print(f"  [{side}] steps={n_sw} cycle={period:.2f}s touches={n_touch} "
          f"lift P50={p50:.1f} P90={p90:.1f}cm gap_mean={mg:.1f}cm t(>3cm)={tf:.0f}%")
    return dict(n=n_sw, cyc=period, touch=n_touch, p50=p50, mg=mg, tf=tf)


seg_len = N // N_SEG
worst = dict(p50=9e9, mg=9e9, tf=9e9, ratio=0, touch=0)
for s in range(N_SEG):
    i0, i1 = s * seg_len, (s + 1) * seg_len - 1
    dur = seg_len * DT
    vel = st.mean(v[i0:i1 + 1])
    cyc = CYC_MIN + CYC_SLOPE * vel if vel > 0.05 else CYC_MIN
    print(f"seg{s} avg_v={vel:.2f} theory_cycle={cyc:.2f}s")
    for side, z, f in (("L", fl, pl), ("R", fr, pr)):
        r = analyze(z, f, side, i0, i1, dur)
        if r["n"] >= 3:
            worst["p50"] = min(worst["p50"], r["p50"])
            worst["mg"] = min(worst["mg"], r["mg"])
            worst["tf"] = min(worst["tf"], r["tf"])
            worst["ratio"] = max(worst["ratio"], r["cyc"] and cyc / r["cyc"])
        worst["touch"] += r["touch"]

ok = (worst["p50"] >= 2.5 and worst["mg"] >= 2.0 and worst["tf"] >= 15
      and worst["ratio"] <= 1.5 and worst["touch"] == 0 and falls == 0)
print("----- VERDICT -----")
print(f"worst P50={worst['p50']:.1f}cm gap={worst['mg']:.1f}cm t(3cm)={worst['tf']:.0f}% "
      f"freq_ratio={worst['ratio']:.2f}x touches={worst['touch']} falls={falls}")
print("ACCEPT" if ok else "REJECT")
