"""Compute cumulative rotation from base_link to left_ankle_roll_link (X1 URDF, zero pose).
URDF rpy: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Chain (parent->child):
  hip_pitch:  rpy=(0, -0.7854, 1.5708)
  hip_roll:   rpy=(1.5708, 0.7854, 0)
  hip_yaw:    rpy=(1.5708, 0, 0)
  knee_pitch: rpy=(-1.5708, 0, -1.5708)
  ankle_pitch:rpy=(-pi, 0, pi)
  ankle_roll: rpy=(0, pi/2, 0)
"""
import numpy as np

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def Rrpy(r, p, y):
    return Rz(y) @ Ry(p) @ Rx(r)

# (rpy, axis, q_default) per joint, parent->child
chain = [
    ((0.0, -0.7854, 1.5708), (0, 0, 1), 0.4),        # left_hip_pitch
    ((1.5707963267949, 0.785398163397451, 0.0), (0, 0, -1), 0.05),  # left_hip_roll
    ((1.5707963267949, 0.0, 0.0), (0, 0, -1), -0.31),  # left_hip_yaw
    ((-1.57079632679488, 0.0, -1.5707963267949), (0, 0, 1), 0.49),  # left_knee_pitch
    ((-np.pi, 0.0, np.pi), (0, 0, -1), -0.21),        # left_ankle_pitch
    ((0.0, np.pi / 2, 0.0), (0, 0, 1), 0.0),          # left_ankle_roll
]

def rot_axis(a, q):
    a = np.array(a, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)

def fk(chain):
    R = np.eye(3)
    for rpy, axis, q in chain:
        R = R @ Rrpy(*rpy) @ rot_axis(axis, q)
    return R

left_chain = [
    ((0.0, -0.7854, 1.5708), (0, 0, 1), 0.4),
    ((1.5707963267949, 0.785398163397451, 0.0), (0, 0, -1), 0.05),
    ((1.5707963267949, 0.0, 0.0), (0, 0, -1), -0.31),
    ((-1.57079632679488, 0.0, -1.5707963267949), (0, 0, 1), 0.49),
    ((-np.pi, 0.0, np.pi), (0, 0, -1), -0.21),
    ((0.0, np.pi / 2, 0.0), (0, 0, 1), 0.0),
]
right_chain = [
    ((0.0, -0.7854, -1.5708), (0, 0, 1), -0.4),
    ((1.57079632679491, 0.78539816339745, 0.0), (0, 0, 1), -0.05),
    ((-1.57079632680723, 0.0, 0.0), (0, 0, 1), 0.31),
    ((1.5707963267949, 0.0, 1.5707963267949), (0, 0, -1), 0.49),
    ((np.pi, 0.0, 0.0), (0, 0, 1), -0.21),
    ((0.0, np.pi / 2, 0.0), (0, 0, 1), 0.0),
]

for tag, chain in [("LEFT", left_chain), ("RIGHT", right_chain)]:
    R = fk(chain)
    print(f"=== {tag} foot (nominal pose) ===")
    for name, col in zip(["local_x", "local_y", "local_z"], R.T):
        print(f"  foot_{name} in base: {np.round(col, 4)}")
    for axis_name, axis_idx in [("x", 0), ("z", 2)]:
        v = R[:, axis_idx]
        vxy = v[:2] / np.linalg.norm(v[:2])
        yaw = np.arctan2(vxy[1], vxy[0])
        print(f"  foot_local_{axis_name} horizontal yaw: {np.degrees(yaw):.2f} deg")
    print()
