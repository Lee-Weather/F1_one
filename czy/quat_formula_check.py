import numpy as np

def my_rotate_z(q):
    qw, qx, qy, qz = q
    fx = 2.0 * (qx * qz + qw * qy)
    fy = 2.0 * (qy * qz - qw * qx)
    fz = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.array([fx, fy, fz])

def ref_quat_rotate(q, v):
    qw, qx, qy, qz = q
    qvec = np.array([qx, qy, qz])
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (qw * uv + uuv)

rng = np.random.default_rng(0)
max_err = 0.0
for _ in range(10000):
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    a = my_rotate_z(q)
    b = ref_quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    max_err = max(max_err, np.abs(a - b).max())
print(f"max err vs isaacgym-style quat_rotate([0,0,1]): {max_err:.3e}")
assert max_err < 1e-9
print("OK: formula matches (Hamilton convention, v=+z)")
