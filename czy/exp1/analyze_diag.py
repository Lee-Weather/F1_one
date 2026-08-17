# Deep analysis of czy/data/exp1/isaac_diag.csv (exp1 adaptive-cycle playback)
import csv
import numpy as np

CSV_PATH = r"e:\X1\F1_one\F1_one\czy\data\exp1\isaac_diag.csv"
DT = 0.02
CYCLE_SPEED_MAX, CYCLE_T_MIN, CYCLE_T_MAX = 0.6, 0.35, 0.7
CONTACT_N = 1.0
SEGMENTS = [(0, 500, 0.2), (500, 1000, 0.4), (1000, 1500, 0.6),
            (1500, 2000, 0.4), (2000, 2500, 0.2), (2500, 3000, 0.0)]


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        keys = ["command_x", "base_vel_x", "base_vel_y", "base_vel_yaw", "base_height",
                "base_pos_x", "base_pos_y", "base_yaw",
                "foot_z_l", "foot_z_r", "foot_force_l", "foot_force_r",
                "foot_yaw_l", "foot_yaw_r"]
        data = {k: [] for k in keys}
        for row in r:
            for k in keys:
                data[k].append(float(row[k]))
    return {k: np.array(v) for k, v in data.items()}


def stance_mask(force):
    return force > CONTACT_N


def rising_intervals(mask):
    """Return list of (start, end) index intervals where mask True."""
    iv, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            iv.append((s, i))
            s = None
    if s is not None:
        iv.append((s, len(mask)))
    return iv


def cycle_period(mask):
    """Gait cycle period from stance rising edges (one cycle per step per foot)."""
    edges = np.where(mask & ~np.roll(mask, 1))[0]
    if len(edges) < 3:
        return None, len(edges)
    return float(np.median(np.diff(edges[1:])) * DT), len(edges) - 1


def stride_length(period, vel):
    return period * vel if period else None


def straightness(d, start, end):
    """Use recorded world-frame base position; fit a line (SVD principal
    direction) and report lateral deviation + path efficiency."""
    x = d["base_pos_x"][start:end]
    y = d["base_pos_y"][start:end]
    # total path length and net displacement
    seglen = np.hypot(np.diff(x), np.diff(y))
    path_len = seglen.sum()
    net = np.hypot(x[-1] - x[0], y[-1] - y[0])
    # line fit (first principal direction of trajectory)
    pts = np.stack([x, y], axis=1)
    p0 = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - p0, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    lateral = (pts - p0) @ np.array([-direction[1], direction[0]])
    # heading drift: base yaw change over segment (deg)
    yaw0 = d["base_yaw"][start]
    yaw1 = d["base_yaw"][end - 1]
    heading_drift = float(np.degrees(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))))
    line_ang = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return {
        "path_len": float(path_len),
        "net": float(net),
        "efficiency": float(net / path_len) if path_len > 1e-6 else 1.0,
        "max_lateral": float(np.abs(lateral).max()),
        "end_lateral": float(abs(lateral[-1] - lateral[0])),
        "heading_drift": heading_drift,
        "line_deg": line_ang,
    }


def main():
    d = load(CSV_PATH)
    print(f"rows={len(d['command_x'])}, dt={DT}s, duration={len(d['command_x'])*DT:.0f}s\n")
    hdr = (f"{'seg':<3} {'cmd':<4} {'real':<5} {'err%':<5} {'theoT':<5} {'measT_L':<7} "
           f"{'measT_R':<7} {'stride':<7} {'2sup%':<5} {'fly%':<5} {'clr_L':<6} {'clr_R':<6} "
           f"{'fmax':<5} {'yawL°':<6} {'yawR°':<6} {'h_avg':<6} {'h_std':<6} {'vyaw':<5}")
    print(hdr)
    for i, (s, e, cmd) in enumerate(SEGMENTS):
        sl = slice(s, e)
        vel = d["base_vel_x"][sl]
        h = d["base_height"][sl]
        fl, fr = d["foot_force_l"][sl], d["foot_force_r"][sl]
        ml, mr = stance_mask(fl), stance_mask(fr)
        theo = CYCLE_T_MIN + (min(cmd, CYCLE_SPEED_MAX) / CYCLE_SPEED_MAX) * (CYCLE_T_MAX - CYCLE_T_MIN)
        tl, nl = cycle_period(ml)
        tr, nr = cycle_period(mr)
        tcomb = np.nanmean([p for p in (tl, tr) if p])
        stride = stride_length(tcomb, float(np.mean(vel)))
        dsup = float(np.mean(ml & mr) * 100)
        fly = float(np.mean(~ml & ~mr) * 100)
        clrL = float(np.mean(d["foot_z_l"][sl][~ml])) if (~ml).any() else 0.0
        clrR = float(np.mean(d["foot_z_r"][sl][~mr])) if (~mr).any() else 0.0
        fmax = float(np.max(np.maximum(fl, fr)))
        yawL = float(np.max(np.abs(np.degrees(d["foot_yaw_l"][sl]))))
        yawR = float(np.max(np.abs(np.degrees(d["foot_yaw_r"][sl]))))
        err = (np.mean(vel) - cmd) / cmd * 100 if cmd > 0 else 0
        print(f"{i:<3} {cmd:<4.1f} {np.mean(vel):<5.3f} {err:<5.0f} {theo:<5.3f} "
              f"{tl if tl else float('nan'):<7.3f} {tr if tr else float('nan'):<7.3f} "
              f"{stride if stride else float('nan'):<7.3f} {dsup:<5.1f} {fly:<5.1f} "
              f"{clrL:<6.3f} {clrR:<6.3f} {fmax:<5.0f} {yawL:<6.1f} {yawR:<6.1f} "
              f"{np.mean(h):<6.3f} {np.std(h):<6.3f} {np.mean(np.abs(d['base_vel_yaw'][sl])):<5.3f}")

    # Global checks
    print("\n== global ==")
    print(f"base_height overall: mean={d['base_height'].mean():.3f} min={d['base_height'].min():.3f} max={d['base_height'].max():.3f}")
    print(f"lateral drift |vel_y| mean={np.abs(d['base_vel_y']).mean():.3f} max={np.abs(d['base_vel_y']).max():.3f}")
    print(f"yaw rate mean={d['base_vel_yaw'].mean():.3f} max|.|={np.abs(d['base_vel_yaw']).max():.3f}")
    print(f"impact peak force max L={d['foot_force_l'].max():.0f}N R={d['foot_force_r'].max():.0f}N")

    # Straightness check (is the robot walking in a straight line?)
    # NOTE: falling triggers env reset -> root position teleports; such segments
    # are flagged RESET and excluded from straightness verdict.
    print("\n== straightness (per segment, from recorded world-frame base position) ==")
    print(f"{'seg':<4}{'cmd':<5}{'dist':<7}{'net':<7}{'eff':<6}{'maxL':<7}{'endL':<7}{'hdrift°':<8}{'verdict'}")
    for i, (s, e, cmd) in enumerate(SEGMENTS):
        st = straightness(d, s, e)
        # detect reset teleport: single-step jump > 0.5 m
        jumps = np.hypot(np.diff(d["base_pos_x"][s:e]), np.diff(d["base_pos_y"][s:e]))
        n_reset = int((jumps > 0.5).sum())
        if n_reset:
            verdict = f"RESET x{n_reset} (fall+respawn, N/A)"
        else:
            ok = st["end_lateral"] < 0.10 and abs(st["heading_drift"]) < 10 and st["efficiency"] > 0.85
            verdict = "STRAIGHT" if ok else "DEVIATED"
        print(f"{i:<4}{cmd:<5.1f}{st['path_len']:<7.2f}{st['net']:<7.2f}{st['efficiency']:<6.3f}"
              f"{st['max_lateral']:<7.3f}{st['end_lateral']:<7.3f}{st['heading_drift']:<8.1f}"
              f"{verdict}")
    # settling: first 100 steps of each segment velocity rms error
    for i, (s, e, cmd) in enumerate(SEGMENTS):
        if cmd == 0:
            continue
        v = d["base_vel_x"][s:min(s + 100, e)]
        print(f"seg{i} first2s vel: mean={v.mean():.3f} std={v.std():.3f}")


if __name__ == "__main__":
    main()
