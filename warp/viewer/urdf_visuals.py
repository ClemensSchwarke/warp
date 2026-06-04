"""Load URDF ``<visual>`` meshes for rendering, with collapse-aware body mapping.

When the URDF is parsed with ``collapse_fixed_joints=True`` (the default in
diffsimrl), many link names no longer correspond to a body in the Model — they
were merged into a parent via fixed joints. Visual meshes are still defined on
the original (collapsed-away) links and need to be re-anchored to the surviving
parent body, with their transform composed through the fixed-joint chain.

Returns a list of dicts that the OpenGL renderer can register and attach as
instances under the appropriate body.

Notes:
    - Mesh material colors are read from URDF ``<material><color rgba=.../></material>``
      when present. Embedded DAE colors are ignored (the older warp OpenGL
      renderer uses per-instance solid colors, not per-vertex / per-face).
    - Meshes that fail to load (missing file, unsupported format) are skipped.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Optional

import numpy as np

import warp as wp


_DEFAULT_VISUAL_COLOR = (0.75, 0.75, 0.75)

# Warn at most once if trimesh is unavailable, so a headless/cluster run where
# the dependency is missing surfaces a clear message instead of silently
# rendering no visual meshes.
_TRIMESH_WARNED = False


def _parse_xyz(node: Optional[ET.Element], attr: str, default=(0.0, 0.0, 0.0)):
    if node is None:
        return tuple(default)
    val = node.get(attr)
    if val is None:
        return tuple(default)
    parts = val.split()
    return tuple(float(p) for p in parts)


def _parse_transform(origin: Optional[ET.Element], scale: float = 1.0) -> wp.transform:
    """Mirror import_urdf.parse_transform but as a free function."""
    if origin is None:
        return wp.transform()
    xyz = _parse_xyz(origin, "xyz", (0.0, 0.0, 0.0))
    rpy = _parse_xyz(origin, "rpy", (0.0, 0.0, 0.0))
    xyz = tuple(v * scale for v in xyz)
    return wp.transform(xyz, wp.quat_rpy(*rpy))


def _resolve_url_to_path(filename: str, urdf_path: str) -> Optional[str]:
    """Resolve a URDF ``<mesh filename="...">`` into an absolute path.

    Supports relative paths (resolved against the URDF's directory).
    """
    if filename.startswith("http://") or filename.startswith("https://"):
        return None  # unsupported here; would need to download
    if os.path.isabs(filename):
        return filename if os.path.exists(filename) else None
    candidate = os.path.normpath(os.path.join(os.path.dirname(urdf_path), filename))
    return candidate if os.path.exists(candidate) else None


def _build_fixed_chain(urdf_root: ET.Element, scale: float = 1.0) -> dict:
    """Map child_link_name -> (parent_link_name, transform_relative_to_parent).

    Only fixed joints participate; non-fixed joints define body boundaries that
    survive collapse.
    """
    chain: dict = {}
    for joint in urdf_root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        tf = _parse_transform(joint.find("origin"), scale)
        chain[child_el.get("link")] = (parent_el.get("link"), tf)
    return chain


def _resolve_body(link_name: str, chain: dict, body_to_idx: dict):
    """Walk up fixed joints from ``link_name`` until a surviving body is found.

    Returns (body_name, body_idx, cumulative_transform_to_body_frame) or None
    if no surviving ancestor exists in the model (shouldn't happen for valid
    URDFs).
    """
    cur = link_name
    cum_tf = wp.transform()
    visited = set()
    while cur not in body_to_idx:
        if cur in visited or cur not in chain:
            return None
        visited.add(cur)
        parent, tf = chain[cur]
        # cum_tf transforms a point in cur's frame into parent's frame: cum_tf' = tf * cum_tf
        cum_tf = wp.transform_multiply(tf, cum_tf)
        cur = parent
    return cur, body_to_idx[cur], cum_tf


def _load_mesh_with_trimesh(path: str, mesh_scale: np.ndarray):
    """Return list of dicts from a mesh file.

    Each dict has keys:
        ``vertices`` (N, 3) float32
        ``indices``  (M,)  int32 flat
        ``uvs``      (N, 2) float32 or ``None``
        ``texture``  PIL.Image or ``None`` (base color texture)
    """
    try:
        import trimesh
    except ImportError:
        global _TRIMESH_WARNED
        if not _TRIMESH_WARNED:
            print(
                "[urdf_visuals] 'trimesh' is not installed, so URDF <visual> "
                "meshes cannot be loaded and the robot will render only its "
                "(possibly hidden) collision shapes. Install it with "
                "'pip install trimesh' to show the visual meshes."
            )
            _TRIMESH_WARNED = True
        return []
    try:
        m = trimesh.load_mesh(path)
    except Exception:
        return []

    geoms = list(m.geometry.values()) if hasattr(m, "geometry") else [m]
    out = []
    for g in geoms:
        if not hasattr(g, "vertices") or not hasattr(g, "faces"):
            continue
        verts = np.asarray(g.vertices, dtype=np.float32) * mesh_scale.astype(np.float32)
        faces = np.asarray(g.faces, dtype=np.int32).reshape(-1)
        if len(verts) == 0 or len(faces) == 0:
            continue

        uvs = None
        texture = None
        visual = getattr(g, "visual", None)
        if visual is not None:
            v_uv = getattr(visual, "uv", None)
            if v_uv is not None:
                try:
                    uvs_arr = np.asarray(v_uv, dtype=np.float32).reshape(-1, 2)
                    if len(uvs_arr) == len(verts):
                        # OpenGL convention is y-up in texture space; trimesh
                        # gives the COLLADA convention which matches.
                        uvs = uvs_arr
                except Exception:
                    pass
            material = getattr(visual, "material", None)
            if material is not None:
                tex = getattr(material, "baseColorTexture", None)
                if tex is not None:
                    texture = tex

        out.append(
            {
                "vertices": verts,
                "indices": faces,
                "uvs": uvs,
                "texture": texture,
            }
        )
    return out


def load_urdf_visuals(
    urdf_path: str,
    body_names: List[str],
    scale: float = 1.0,
    default_color=_DEFAULT_VISUAL_COLOR,
) -> List[dict]:
    """Parse ``<visual>`` mesh entries from a URDF and return one entry per geometry.

    Args:
        urdf_path: Absolute path to the URDF file.
        body_names: ``model.body_name`` after URDF parsing + fixed-joint collapse.
            Used to identify surviving bodies.
        scale: URDF scale (same as passed to ``parse_urdf``).
        default_color: Fallback color when the URDF has no ``<material><color>``.

    Returns:
        A list of dicts, each describing one visual mesh instance:
            ``body_name``: surviving body name (a key in ``body_names``).
            ``body_idx``: index in ``body_names``.
            ``vertices``: (N, 3) float32 array.
            ``indices``: (M,) int32 flat triangle index array.
            ``pos``: (3,) float position in body frame.
            ``rot``: (4,) float quaternion (x, y, z, w) in body frame.
            ``color``: (3,) float RGB.
    """
    if not os.path.exists(urdf_path):
        return []

    root = ET.parse(urdf_path).getroot()
    chain = _build_fixed_chain(root, scale)

    # Build URDF-level material catalog (named materials defined at the top of
    # the URDF and referenced by <material name="..."/> from a link).
    materials: dict = {}
    for mat in root.findall("material"):
        name = mat.get("name")
        color_el = mat.find("color")
        if name and color_el is not None:
            rgba = (color_el.get("rgba") or "").split()
            if len(rgba) >= 3:
                materials[name] = tuple(float(x) for x in rgba[:3])

    body_to_idx = {n: i for i, n in enumerate(body_names)}

    entries: List[dict] = []
    for urdf_link in root.findall("link"):
        link_name = urdf_link.get("name")
        if link_name is None:
            continue
        for visual in urdf_link.findall("visual"):
            geo = visual.find("geometry")
            if geo is None:
                continue
            mesh_el = geo.find("mesh")
            if mesh_el is None:
                continue
            filename = mesh_el.get("filename")
            if not filename:
                continue
            path = _resolve_url_to_path(filename, urdf_path)
            if path is None:
                continue

            mesh_scale_str = mesh_el.get("scale") or "1 1 1"
            mesh_scale = np.array(
                [float(x) * scale for x in mesh_scale_str.split()], dtype=np.float32
            )

            visual_tf = _parse_transform(visual.find("origin"), scale)

            # color: <material> inline color or named reference
            color = default_color
            mat_el = visual.find("material")
            if mat_el is not None:
                inline = mat_el.find("color")
                if inline is not None:
                    rgba = (inline.get("rgba") or "").split()
                    if len(rgba) >= 3:
                        color = tuple(float(x) for x in rgba[:3])
                elif mat_el.get("name") in materials:
                    color = materials[mat_el.get("name")]

            resolved = _resolve_body(link_name, chain, body_to_idx)
            if resolved is None:
                continue
            _body_name, body_idx, link_to_body_tf = resolved

            # Transform from visual frame -> body frame:
            #   x_body = link_to_body_tf * (visual_tf * x_visual)
            tf = wp.transform_multiply(link_to_body_tf, visual_tf)

            geom_data = _load_mesh_with_trimesh(path, mesh_scale)
            for geom in geom_data:
                entries.append(
                    {
                        "body_name": _body_name,
                        "body_idx": body_idx,
                        "vertices": geom["vertices"],
                        "indices": geom["indices"],
                        "uvs": geom["uvs"],
                        "texture": geom["texture"],
                        "pos": tuple(float(v) for v in tf.p),
                        "rot": tuple(float(v) for v in tf.q),
                        "color": tuple(color),
                    }
                )

    return entries
