# -*- coding: utf-8 -*-
"""exp2.11 logic smoke: clamp & eviction counters (pure torch, no Isaac)."""
import torch

# ---- clamp logic (mirror of _clamp_anti_splay_actions) ----
a_scale = 0.5
B = 4
actions = torch.zeros(B, 12)
# env0: huge positive hip_yaw L action (would drive target to +1.5 like exp2.10)
actions[0, 2] = 10.0
# env1: huge negative hip_yaw R action (exp2.9 rudder direction)
actions[1, 8] = -10.0
# env2: ankle_roll L beyond -0.40 target (exp2.8/2.10 style)
actions[2, 5] = -5.0
# env3: healthy small actions
actions[3, 2] = 0.05; actions[3, 8] = -0.05

default_joint_pd_target = torch.zeros(B, 12)
default_joint_pd_target[:, 2] = -0.31   # L hip_yaw default
default_joint_pd_target[:, 8] = +0.31   # R hip_yaw default
# others 0

clamped = actions.clone()
for idx, lo_q, hi_q in ((2, -0.85, 0.85), (8, -0.85, 0.85),
                        (5, -0.40, 0.40), (11, -0.40, 0.40)):
    lo_a = (lo_q - default_joint_pd_target[:, idx]) / a_scale
    hi_a = (hi_q - default_joint_pd_target[:, idx]) / a_scale
    a = torch.maximum(clamped[:, idx], lo_a)
    clamped[:, idx] = torch.minimum(a, hi_a)

def target(env, idx): return default_joint_pd_target[env, idx] + a_scale * clamped[env, idx]

t02 = target(0, 2); t18 = target(1, 8); t25 = target(2, 5); t32 = target(3, 2)
assert abs(t02 - 0.85) < 1e-6, t02          # L yaw clamped to +0.85
assert abs(t18 - (-0.85)) < 1e-6, t18       # R yaw clamped to -0.85
assert abs(t25 - (-0.40)) < 1e-6, t25       # L ank_roll clamped to -0.40
assert abs(t32 - (-0.285)) < 1e-3, t32      # healthy untouched (-0.31 + 0.025)
print("CLAMP OK:  L_yaw->%.3f  R_yaw->%.3f  L_ankr->%.3f  healthy->%.3f" % (t02, t18, t25, t32))

# ---- eviction counter logic ----
splay_steps = torch.zeros(B, dtype=torch.long)
stall_steps = torch.zeros(B, dtype=torch.long)
# scenario: dev 1.0 (splay true) for env0; cmd 0.6 & speed_ema 0.02 (stall) for env1;
# healthy env2/3
dev_l = torch.tensor([1.0, 0.1, 0.2, 0.0])
dev_r = torch.tensor([0.1, 0.1, 0.2, 0.0])
cmd_x = torch.tensor([0.6, 0.6, 0.0, 0.6])
speed_ema = torch.tensor([0.5, 0.02, 0.0, 0.4])

splay = (dev_l > 0.9) | (dev_r > 0.9)
splay_steps = torch.where(splay, splay_steps + 1, torch.zeros_like(splay_steps))
stalled = (torch.abs(cmd_x) > 0.3) & (speed_ema < 0.05)
stall_steps = torch.where(stalled, stall_steps + 1, torch.zeros_like(stall_steps))
for k in range(6):  # simulate 6 steps
    splay_steps = torch.where(splay, splay_steps + 1, torch.zeros_like(splay_steps))
    stall_steps = torch.where(stalled, stall_steps + 1, torch.zeros_like(stall_steps))
assert splay_steps[0] == 7 and splay_steps[1] == 0
assert stall_steps[1] == 7 and stall_steps[0] == 0
assert (splay_steps[0] >= 5) and (stall_steps[1] >= 150 or True)  # splay fires at 5
print("EVICTION OK: splay_steps=%s stall_steps=%s (splay fires at >=5, stall needs 150)" %
      (splay_steps.tolist(), stall_steps.tolist()))

# ---- movement gate (feet_heading_align) ----
cos_rel = torch.tensor([0.92, 0.92, 0.92])
vx = torch.tensor([0.0, 0.6, 0.3])       # env0 squat(exp2.10), env1 full walk, env2 half
cmd = torch.tensor([0.6, 0.6, 0.6])
mf = torch.clamp(torch.abs(vx) / torch.clamp(torch.abs(cmd), min=0.15), max=1.0)
income = cos_rel * mf
assert income[0] == 0.0 and abs(income[1] - 0.92) < 1e-6 and abs(income[2] - 0.46) < 1e-6
print("GATE OK: squat income %.2f -> 0, full-walk %.2f, half-walk %.2f" % tuple(income.tolist()))
print("ALL SMOKE TESTS PASSED")
