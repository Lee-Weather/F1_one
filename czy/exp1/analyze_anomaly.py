# Locate anomalies in isaac_diag.csv: height dips, foot yaw swings, vel spikes
import csv
import numpy as np

CSV_PATH = r"e:\X1\F1_one\F1_one\czy\data\exp1\isaac_diag.csv"
DT = 0.02
SEGMENTS = [(0, 500, 0.2), (500, 1000, 0.4), (1000, 1500, 0.6),
            (1500, 2000, 0.4), (2000, 2500, 0.2), (2500, 3000, 0.0)]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        keys = ["command_x", "base_vel_x", "base_vel_y", "base_height",
                "foot_yaw_l", "foot_yaw_r", "foot_force_l", "foot_force_r"]
        data = {k: [] for k in keys}
        for row in r:
            for k in keys:
                data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}


def seg_of(i):
    for n, (s, e, c) in enumerate(SEGMENTS):
        if s <= i < e:
            return n, c
    return -1, 0


def main():
    d = load(CSV_PATH)
    h = d["base_height"]
    # 1) height below 0.4 (dip/crouch)
    low = np.where(h < 0.45)[0]
    print(f"height<0.45m steps: {len(low)}")
    if len(low):
        # cluster
        cl = []
        s = low[0]
        for a, b in zip(low, low[1:]):
            if b - a > 5:
                cl.append((s, a))
                s = b
        cl.append((s, low[-1]))
        for s, e in cl:
            print(f"  t={s*DT:.1f}s~{e*DT:.1f}s (seg{seg_of(s)[0]}, cmd={seg_of(s)[1]}) min_h={h[s:e+1].min():.3f}")

    # 2) foot yaw > 30 deg
    for name in ["foot_yaw_l", "foot_yaw_r"]:
        yaw = np.degrees(d[name])
        big = np.where(np.abs(yaw) > 30)[0]
        if len(big):
            t0, t1 = big[0] * DT, big[-1] * DT
            print(f"{name} |>30deg| steps: {len(big)}, t={t0:.1f}s~{t1:.1f}s (seg{seg_of(big[0])[0]}), max={np.abs(yaw[big]).max():.0f}deg")

    # 3) lateral velocity spikes
    vy = d["base_vel_y"]
    sp = np.where(np.abs(vy) > 0.5)[0]
    print(f"|vel_y|>0.5 steps: {len(sp)}", end="")
    if len(sp):
        print(f", t={sp[0]*DT:.1f}s~{sp[-1]*DT:.1f}s (seg{seg_of(sp[0])[0]})")
    else:
        print()

    # 4) per-segment settling time to reach 90% of cmd
    for n, (s, e, c) in enumerate(SEGMENTS):
        if c == 0:
            continue
        v = d["base_vel_x"][s:e]
        target = np.full(len(v), c)
        # skip first 1s transient, report error in last 5s
        v_late = v[-250:]
        err = np.mean(np.abs(v_late - c))
        print(f"seg{n} cmd={c}: late5s mean|x_err|={err:.3f} m/s, mean_v={v_late.mean():.3f}")

    # 5) stance force stats at 0.6 segment (impact severity)
    for n, (s, e, c) in enumerate(SEGMENTS[:3]):
        fl, fr = d["foot_force_l"][s:e], d["foot_force_r"][s:e]
        con = np.maximum(fl, fr) > 1.0
        if con.any():
            fmax_l = np.percentile(fl[fl > 1], 95) if (fl > 1).any() else 0
            fmax_r = np.percentile(fr[fr > 1], 95) if (fr > 1).any() else 0
            print(f"seg{n}: stance force p95 L={fmax_l:.0f}N R={fmax_r:.0f}N")


if __name__ == "__main__":
    main()
