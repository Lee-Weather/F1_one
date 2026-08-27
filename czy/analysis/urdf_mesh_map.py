# -*- coding: utf-8 -*-
"""新 URDF mesh 引用 → 现有 STL 语义映射。"""
import os
import re
import xml.etree.ElementTree as ET

NEW = "resources/robots/x1/urdf/X1_12DOF_physically_mirrored.urdf"
OLD = "resources/robots/x1/urdf/X1_12DOF.urdf"
MESHD = "resources/robots/x1/meshes"

have = set(f for f in os.listdir(MESHD) if f.endswith(".STL"))

def refs(path):
    """[(link, tag(visual/collision), meshfile)]"""
    out = []
    for L in ET.parse(path).getroot().findall("link"):
        for tag in ("visual", "collision"):
            g = L.find(f"{tag}/geometry/mesh")
            if g is not None:
                out.append((L.get("name"), tag, os.path.basename(g.get("filename"))))
    return out

new_refs = refs(NEW)
old_refs = refs(OLD)

def candidates(name):
    """按规则生成候选现有文件名（优先级序）。"""
    c = []
    base = name[:-4]  # strip .STL
    for suf in ("_physically_mirrored", "_center_symmetric_collision", "_center_symmetric"):
        if base.endswith(suf):
            c.append(base[: -len(suf)] + ".STL")
    # right_ 系列: arm_r_wrist_*_right_physically_mirrored
    m = re.match(r"(.+)_right_physically_mirrored$", base)
    if m:
        c.append(m.group(1) + ".STL")
        c.append(m.group(1) + "_right.STL")
    # 特例: lumbar_pitch_x1 -> lumbar_pitch (x1 后缀从未有文件)
    if base.startswith("lumbar_pitch"):
        c.append("lumbar_pitch.STL")
    if base.startswith("base_link_simple"):
        c.append("base_link_simple.STL")
    # collision 特例
    if base.endswith("_collision"):
        c.append(base + ".STL")
    return c

print(f"{'link':<32}{'tag':<11}{'新引用':<58}{'映射到'}")
print("-" * 130)
resolved, unresolved = [], []
seen = set()
for link, tag, mesh in new_refs:
    if (link, tag, mesh) in seen:
        continue
    seen.add((link, tag, mesh))
    if mesh in have:
        resolved.append((link, tag, mesh, mesh, "直接存在"))
        print(f"{link:<32}{tag:<11}{mesh:<58}{mesh}  [直接存在]")
        continue
    for c in candidates(mesh):
        if c in have:
            resolved.append((link, tag, mesh, c, "规则映射"))
            print(f"{link:<32}{tag:<11}{mesh:<58}{c}  [规则映射]")
            break
    else:
        unresolved.append((link, tag, mesh))
        print(f"{link:<32}{tag:<11}{mesh:<58}**无候选**")

print()
print(f"解析: {len(resolved)} 项 / 未解析: {len(unresolved)} 项")

# 老URDF里对应的mesh(对比列)
old_map = {(l, t): m for l, t, m in old_refs}
print()
print("=" * 130)
print("新旧 mesh 引用对照（同一 link+tag）")
print("=" * 130)
for link, tag, mesh, target, how in resolved:
    oldm = old_map.get((link, tag), "-")
    mark = "" if oldm == target else "  <- 与旧不同" + (f" (旧:{oldm})" if oldm != "-" else "")
    print(f"{link:<30}{tag:<10}{oldm:<45}{target}{mark}")
