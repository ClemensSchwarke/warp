"""OpenGL viewer backend (pyglet) for warp 1.0.0-beta.2.

Thin wrapper around ``wp.sim.render.SimRendererOpenGL``. Reuses the existing
pyglet renderer's shape registration, instancing, camera callbacks, and ground
rendering, with three improvements aimed at producing newton-like output:

1. **Neutral collision color.** The base ``SimRenderer`` assigns a tab10 color
   per shape (rainbow palette), which makes a 44-body G1 look "all over the
   place". We override ``_get_new_color`` to return a single soft grey so
   collision shapes recede visually under the URDF visual meshes.

2. **URDF visual meshes.** Optionally accepts a ``urdf_path`` and uses
   :mod:`warp.viewer.urdf_visuals` to parse ``<visual>`` mesh tags. Visuals
   are re-anchored to their surviving body after ``collapse_fixed_joints``,
   registered as new shapes, and attached as instances — so they update via
   the same ``update_body_transforms`` pipeline as the collision shapes.

3. **Dark grey terrain.** Rough terrain is rendered as a single dark grey
   matte mesh (matches newton's ground feel).

Contact-normal arrows and a non-blocking ``save()`` are also provided.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from .viewer import ViewerBase
from .urdf_visuals import load_urdf_visuals


# Newton-inspired neutral palette
_COLLISION_COLOR = (0.78, 0.78, 0.80)  # warm grey for collision shapes
# Darker terrain albedo so the shader's stacked (ambient + sun diffuse + fill
# diffuse + specular) lighting does not saturate to white on flat tops, which
# is what flattens the visible roughness. With albedo ~0.30 the brightest
# surfaces sit around 0.55-0.65 while shadow-facing surfaces drop to ~0.10,
# producing the high-contrast matte look IsaacLab achieves with its raked sun.
_TERRAIN_COLOR = (0.30, 0.30, 0.32)
_CONTACT_COLOR = (1.00, 0.40, 0.00)    # orange
_NORMAL_COLOR = (1.00, 0.10, 0.10)     # red (contact normal arrows)

# Sky / ground colors (modeled on newton viewer's neutral palette).
_SKY_COLOR = (0.22, 0.24, 0.28)        # dark blue-grey background
_SKY_HAZE = (0.40, 0.42, 0.48)         # lighter neutral haze near the horizon
_GROUND_TILE_A = (0.38, 0.40, 0.43)    # darker tile
_GROUND_TILE_B = (0.30, 0.32, 0.34)    # secondary tile

# Sun direction. The base renderer ships with a near-vertical sun (-0.2, 0.8,
# 0.3) which produces uniform high diffuse on almost-horizontal surfaces — bad
# for showing terrain roughness. We tilt the sun to a more raking angle so the
# cosine term swings more across small surface tilts, matching IsaacLab's
# dramatic side-lit look on terrain bumps. (X is the bake's lateral axis; Y is
# up; Z is the bake's forward axis.)
_SUN_DIRECTION = (-0.5, 0.55, 0.4)


# NOTE: the contact-filter Warp kernel lives in ``contact_filter.py`` instead
# of this module. ``from __future__ import annotations`` (top of this file)
# stringifies type annotations, which breaks Warp's kernel signature parser —
# it fails to recognise ``wp.array(...)`` as an array type and raises
# "Can only subscript assign array, vector, and matrix types" on the first
# ``out[tid] = ...`` line. The sibling module deliberately omits that future
# import so kernel parameter types stay as real objects.
from .contact_filter import filter_ground_contacts_kernel as _filter_ground_contacts_kernel


class _NeutralSimRendererOpenGL:
    """Build-time mixin: produces a ``SimRendererOpenGL`` subclass that overrides
    ``_get_new_color`` to a single neutral grey, so collision shapes don't cycle
    through the tab10 palette.

    Built lazily so warp.sim.render isn't imported at module top level.
    """

    _cls = None

    @classmethod
    def make(cls):
        if cls._cls is None:

            class _NeutralRenderer(warp.sim.render.SimRendererOpenGL):
                def _get_new_color(self):
                    return _COLLISION_COLOR

            cls._cls = _NeutralRenderer
        return cls._cls


class ViewerGL(ViewerBase):
    """Live OpenGL viewer.

    Per-frame call sequence:
        ``begin_frame(t) → log_state(state) [→ log_terrain(...) once] → end_frame()``.
    Camera control (from the underlying pyglet renderer):
        left-mouse-drag to rotate, scroll to zoom, WASD to translate, ESC to
        close, SPACE to pause.
    """

    def __init__(
        self,
        model: warp.sim.Model,
        title: str = "warp viewer",
        fps: int = 60,
        up_axis: str = "Y",
        show_contacts: bool = True,
        show_normals: bool = True,
        show_joints: bool = False,
        screen_width: int = 1280,
        screen_height: int = 720,
        scaling: float = 1.0,
        contact_points_radius: float = 0.02,
        contact_normal_length: float = 0.15,
        contact_normal_radius: float = 0.006,
        contact_activation_dist: float = 1.0e-3,
        contact_dedup_radius: float = 0.015,
        urdf_path: Optional[str] = None,
        urdf_scale: float = 1.0,
        hide_collision_shapes: bool = False,
    ):
        super().__init__()
        self.model = model
        # User-toggleable display flags. The base SimRenderer's built-in
        # contact spheres (which draw all broadphase candidates as two spheres
        # per pair) are disabled below — we replace them with a filtered
        # one-sphere-per-actual-ground-contact view.
        self.show_contacts = show_contacts
        self.show_normals = show_normals
        self._contact_points_radius = contact_points_radius
        self._contact_normal_length = contact_normal_length
        self._contact_normal_radius = contact_normal_radius
        self._contact_activation_dist = contact_activation_dist
        self._contact_dedup_radius = contact_dedup_radius

        # Build a SimRendererOpenGL subclass that uses a neutral colour for
        # collision shapes instead of cycling tab10. SimRendererOpenGL forwards
        # `path` to `OpenGLRenderer.__init__(title=path, ...)`.
        renderer_cls = _NeutralSimRendererOpenGL.make()
        self._sim_renderer = renderer_cls(
            model=model,
            path=title,
            scaling=scaling,
            fps=fps,
            up_axis=up_axis,
            # We always disable the base SimRenderer's broadphase contact
            # spheres — it draws all candidate pairs (two spheres each) whether
            # in contact or not. Our own filtered pipeline runs in log_state()
            # below and emits one sphere per *active* ground contact.
            show_rigid_contact_points=False,
            contact_points_radius=contact_points_radius,
            show_joints=show_joints,
            screen_width=screen_width,
            screen_height=screen_height,
            background_color=_SKY_COLOR,
        )
        # The base OpenGLRenderer hard-codes the sky shader's haze color to a
        # warm orange. Override it to a neutral horizon tint so the scene
        # reads as Newton-like rather than sunset.
        self._patch_sky_haze(_SKY_HAZE)
        # Tilt the sun to a raking angle so terrain roughness reads via
        # cos-shading variation instead of saturating to white on flat tops.
        self._patch_sun_direction(_SUN_DIRECTION)

        # Re-color the ground plane with Newton-like darker tiles (the base
        # renderer's render_ground bakes in light grey 200/150 tones).
        self._recolor_ground(_GROUND_TILE_A, _GROUND_TILE_B)

        # Load and register URDF visual meshes.
        num_visuals = 0
        if urdf_path is not None:
            num_visuals = self._register_urdf_visuals(urdf_path, urdf_scale)

        # Optionally hide collision shapes by scaling them to near-zero so the
        # visual meshes don't fight with them for pixels. Only do this when
        # visual meshes actually loaded: if they didn't (e.g. trimesh missing
        # on a headless cluster, or the mesh files are absent), hiding the
        # collision proxies would leave the robot completely invisible. In that
        # case keep the collision primitives so the robot is still shown.
        if hide_collision_shapes:
            if num_visuals > 0:
                self._hide_collision_shapes()
            else:
                print(
                    "[ViewerGL] No URDF visual meshes were loaded from "
                    f"{urdf_path!r}; keeping collision shapes visible so the "
                    "robot is not invisible. Install 'trimesh' and ensure the "
                    "URDF's mesh files exist to render the visual meshes."
                )

        # Persistent device buffers for the filtered ground-contact pipeline.
        # Sized to rigid_contact_max so we can launch one thread per broadphase
        # slot; inactive slots are written as a sentinel and dropped on host.
        rcm = int(getattr(model, "rigid_contact_max", 0) or 0)
        self._filter_max = rcm
        if rcm > 0:
            self._filtered_pos = wp.zeros(rcm, dtype=wp.vec3, device=model.device)
            self._filtered_normal = wp.zeros(rcm, dtype=wp.vec3, device=model.device)
            self._filtered_body = wp.zeros(rcm, dtype=wp.int32, device=model.device)
        else:
            self._filtered_pos = None
            self._filtered_normal = None
            self._filtered_body = None

        # Register a key handler for toggle keys. Pyglet supports multiple
        # on_key_press handlers via push_handlers; the base renderer already
        # registered its own (ESC/SPACE/...), so this one runs alongside.
        self._sim_renderer.window.push_handlers(on_key_press=self._on_key_press)

    # ------------------------------------------------------------------
    # frame lifecycle
    # ------------------------------------------------------------------
    def begin_frame(self, time: float) -> None:
        self.time = time
        self._sim_renderer.begin_frame(time)

    def end_frame(self) -> None:
        self._sim_renderer.end_frame()

    # ------------------------------------------------------------------
    # logging methods
    # ------------------------------------------------------------------
    def log_state(self, state: warp.sim.State) -> None:
        self._sim_renderer.render(state)
        self._render_filtered_ground_contacts(state)

    def log_terrain(self, vertices, indices, name: str = "terrain") -> None:
        if self._terrain_logged:
            return
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        idx = np.asarray(indices, dtype=np.int32).reshape(-1)
        self._sim_renderer.render_mesh(
            name=name,
            points=verts,
            indices=idx,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
            colors=[_TERRAIN_COLOR] * len(verts),
        )
        # The OpenGLRenderer stores its checkerboard colors on the registered
        # shape. Override both color1 and color2 to the same dark grey so the
        # terrain reads as a solid matte surface (no rainbow checker pattern).
        self._force_solid_shape_color(name, _TERRAIN_COLOR)
        self._terrain_logged = True

    # ------------------------------------------------------------------
    # URDF visuals
    # ------------------------------------------------------------------
    def _register_urdf_visuals(self, urdf_path: str, urdf_scale: float) -> int:
        """Register URDF ``<visual>`` meshes as instances. Returns the number of
        visual mesh instances actually registered (0 if none loaded)."""
        body_names = list(getattr(self.model, "body_name", []))
        if not body_names:
            return 0
        # The body_name list on `model` repeats across envs (one set of names
        # per env). `SimRenderer.populate` namespaces body names with index
        # prefixes so they collide-free across envs. To attach visuals only to
        # the first env (visualisation purposes), we use the first
        # `bodies_per_env` entries.
        bodies_per_env = max(1, self.model.body_count // max(1, self.model.num_envs))
        first_env_body_names = body_names[:bodies_per_env]

        entries = load_urdf_visuals(urdf_path, first_env_body_names, scale=urdf_scale)
        if not entries:
            return 0

        # SimRenderer.populate names bodies as f"body_{b}_{model.body_name[b]}".
        # Look up that prefixed name so `add_shape_instance`'s body resolution
        # picks the right parent.
        for entry in entries:
            b = entry["body_idx"]
            prefixed_body_name = f"body_{b}_{entry['body_name'].replace(' ', '_')}"
            self._add_visual_mesh(
                name=f"visual_{b}_{id(entry)}",
                vertices=entry["vertices"],
                indices=entry["indices"],
                pos=entry["pos"],
                rot=entry["rot"],
                parent_body=prefixed_body_name,
                color=entry["color"],
                uvs=entry.get("uvs"),
                texture=entry.get("texture"),
            )
        return len(entries)

    def _add_visual_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        pos: tuple,
        rot: tuple,
        parent_body: str,
        color: tuple,
        uvs=None,
        texture=None,
    ) -> None:
        # Register as a template (geometry only) so we can pin per-instance
        # colors via add_shape_instance. When a texture is provided it overrides
        # the per-instance color via the new useAlbedoTex shader path.
        shape = self._sim_renderer.render_mesh(
            name=name,
            points=vertices,
            indices=indices,
            pos=pos,
            rot=rot,
            scale=(1.0, 1.0, 1.0),
            colors=[color] * len(vertices),
            parent_body=parent_body,
            is_template=True,
            uvs=uvs,
            texture_image=texture,
        )
        # Pin both checker colors to the same value so the surface reads as
        # solid (no white/black moiré bands) when no texture is bound. This is
        # a no-op for textured shapes since the shader picks the texture path.
        self._sim_renderer.add_shape_instance(
            name,
            shape,
            parent_body,
            pos,
            rot,
            (1.0, 1.0, 1.0),
            color1=color,
            color2=color,
        )

    # ------------------------------------------------------------------
    # filtered ground-contact rendering
    # ------------------------------------------------------------------
    _CONTACT_SPHERE_INSTANCER = "ground_contact_points"
    _CONTACT_NORMAL_INSTANCER = "ground_contact_normals"

    def _render_filtered_ground_contacts(self, state: warp.sim.State) -> None:
        """Filter the broadphase contact buffer to actual ground contacts and
        render one sphere + one red normal arrow per contact. Skips work for
        whichever toggle is off, and removes any stale instancer so toggling
        off hides the geometry immediately.
        """
        if not self.show_contacts and not self.show_normals:
            self._drop_instancer(self._CONTACT_SPHERE_INSTANCER)
            self._drop_instancer(self._CONTACT_NORMAL_INSTANCER)
            return
        if self._filter_max <= 0:
            return
        model = self.model
        if model is None or not getattr(model, "rigid_contact_max", 0):
            return

        wp.launch(
            kernel=_filter_ground_contacts_kernel,
            dim=self._filter_max,
            inputs=[
                state.body_q,
                model.shape_body,
                model.shape_geo.thickness,
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                float(self._contact_activation_dist),
            ],
            outputs=[self._filtered_pos, self._filtered_normal, self._filtered_body],
            device=model.device,
        )

        pos = self._filtered_pos.numpy()
        nrm = self._filtered_normal.numpy()
        bod = self._filtered_body.numpy()
        # Sentinel-filter (kernel writes -1e8 / -1 for inactive slots).
        valid = (pos[:, 1] > -1.0e7) & (bod >= 0)
        pos = pos[valid]
        nrm = nrm[valid]
        bod = bod[valid]

        # Deduplicate by *spatial location*: the broadphase occasionally
        # emits multiple slots at the same contact point (anymal point feet
        # see this), but G1's foot legitimately produces several contacts at
        # distinct corners of the same body. Per-body dedup would collapse
        # the G1 case; greedy spatial clustering within
        # ``contact_dedup_radius`` only merges contacts that share a body
        # AND sit on top of each other, which is what we want.
        if len(pos) > 1 and self._contact_dedup_radius > 0.0:
            r2 = self._contact_dedup_radius * self._contact_dedup_radius
            keep_idx: list[int] = []
            kept_pos: list[np.ndarray] = []
            kept_bod: list[int] = []
            for i in range(len(pos)):
                p_i = pos[i]
                b_i = int(bod[i])
                duplicate = False
                for kp, kb in zip(kept_pos, kept_bod):
                    if kb != b_i:
                        continue
                    d = p_i - kp
                    if float(d @ d) <= r2:
                        duplicate = True
                        break
                if not duplicate:
                    keep_idx.append(i)
                    kept_pos.append(p_i)
                    kept_bod.append(b_i)
            keep_arr = np.asarray(keep_idx, dtype=np.int64)
            pos = pos[keep_arr]
            nrm = nrm[keep_arr]

        if self.show_contacts:
            if len(pos) > 0:
                self._sim_renderer.render_points(
                    name=self._CONTACT_SPHERE_INSTANCER,
                    points=np.ascontiguousarray(pos, dtype=np.float32),
                    radius=self._contact_points_radius,
                    colors=[_CONTACT_COLOR] * len(pos),
                )
            else:
                self._drop_instancer(self._CONTACT_SPHERE_INSTANCER)
        else:
            self._drop_instancer(self._CONTACT_SPHERE_INSTANCER)

        if self.show_normals:
            if len(pos) > 0:
                verts, idx = self._build_arrow_segments(pos, nrm, self._contact_normal_length)
                self._sim_renderer.render_line_list(
                    name=self._CONTACT_NORMAL_INSTANCER,
                    vertices=verts,
                    indices=idx,
                    color=_NORMAL_COLOR,
                    radius=self._contact_normal_radius,
                )
            else:
                self._drop_instancer(self._CONTACT_NORMAL_INSTANCER)
        else:
            self._drop_instancer(self._CONTACT_NORMAL_INSTANCER)

    @staticmethod
    def _build_arrow_segments(
        positions: np.ndarray, normals: np.ndarray, length: float
    ) -> tuple:
        """Build a vertex/index buffer drawing an arrow per (position, normal):
        a shaft segment plus two head segments forming a small "<" at the tip.
        Returns (vertices [3N*4, 3], indices [3N*2]).
        """
        n = len(positions)
        # Renormalize to unit length (defensive — kernel emits a unit normal).
        mag = np.linalg.norm(normals, axis=1, keepdims=True)
        mag[mag < 1.0e-8] = 1.0
        d = normals / mag

        # Build a perpendicular vector per contact (Hughes-Möller style choice).
        ref = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        flip = np.abs(d[:, 0]) > 0.9
        if flip.any():
            ref[flip] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        perp = np.cross(d, ref)
        perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1.0e-8)

        head_len = 0.04
        head_w = 0.03
        tip = positions + d * length
        base = positions + d * (length - head_len)
        h1 = base + perp * head_w
        h2 = base - perp * head_w

        verts = np.empty((4 * n, 3), dtype=np.float32)
        verts[0::4] = positions
        verts[1::4] = tip
        verts[2::4] = h1
        verts[3::4] = h2
        # Three line segments per arrow: shaft, tip→h1, tip→h2.
        idx = np.empty((n, 6), dtype=np.int32)
        base_idx = (np.arange(n, dtype=np.int32) * 4)[:, None]
        idx[:, 0:2] = base_idx + np.array([0, 1], dtype=np.int32)
        idx[:, 2:4] = base_idx + np.array([1, 2], dtype=np.int32)
        idx[:, 4:6] = base_idx + np.array([1, 3], dtype=np.int32)
        return verts, idx.reshape(-1)

    def _drop_instancer(self, name: str) -> None:
        """Remove a previously-created instancer so it stops drawing.

        ``render_points`` / ``render_line_list`` cache a ``ShapeInstancer`` in
        ``OpenGLRenderer._shape_instancers``; the per-frame draw loop iterates
        that dict, so popping the entry hides the geometry immediately.
        """
        renderer = self._sim_renderer
        instancers = getattr(renderer, "_shape_instancers", None)
        if instancers is None:
            return
        instancers.pop(name, None)

    def _on_key_press(self, symbol, modifiers):
        """Toggle filtered-contact display ('P') and normal arrows ('N')."""
        import pyglet
        if symbol == pyglet.window.key.P:
            self.show_contacts = not self.show_contacts
            if not self.show_contacts:
                self._drop_instancer(self._CONTACT_SPHERE_INSTANCER)
        elif symbol == pyglet.window.key.N:
            self.show_normals = not self.show_normals
            if not self.show_normals:
                self._drop_instancer(self._CONTACT_NORMAL_INSTANCER)

    # ------------------------------------------------------------------
    # contact rendering
    # ------------------------------------------------------------------
    def _render_contact_points(self, positions: np.ndarray, radius: float) -> None:
        n = len(positions)
        self._sim_renderer.render_points(
            name="contacts",
            points=positions,
            radius=radius,
            colors=[_CONTACT_COLOR] * n,
        )

    def _render_contact_normals(self, starts: np.ndarray, ends: np.ndarray) -> None:
        n = len(starts)
        if n == 0:
            return
        verts = np.empty((2 * n, 3), dtype=np.float32)
        verts[0::2] = starts
        verts[1::2] = ends
        idx = np.arange(2 * n, dtype=np.int32)
        self._sim_renderer.render_line_list(
            name="contact_normals",
            vertices=verts,
            indices=idx,
            color=_NORMAL_COLOR,
            radius=0.005,
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _force_solid_shape_color(self, instance_name: str, color: tuple) -> None:
        """Set color1 and color2 of an existing instance to the same value.

        The OpenGLRenderer stores per-shape default colors in ``_shapes`` and
        per-instance overrides in ``_instances[name]``. The fragment shader
        blends color1/color2 by a checker pattern; equal colors → flat shade.
        """
        renderer = self._sim_renderer
        if instance_name not in renderer._instances:
            return
        instance_id, body, shape, transform, scale, _c1, _c2 = renderer._instances[instance_name]
        renderer._instances[instance_name] = (
            instance_id,
            body,
            shape,
            transform,
            scale,
            color,
            color,
        )
        # Also update the stored shape defaults so any future instance picks them up.
        if shape < len(renderer._shapes):
            verts, idx_arr, _c1, _c2, geo_hash = renderer._shapes[shape]
            renderer._shapes[shape] = (verts, idx_arr, color, color, geo_hash)
        renderer._add_shape_instances = True

    def _hide_collision_shapes(self) -> None:
        """Make all currently-registered collision shapes transparent-ish by
        moving them to a near-zero scale. Useful when URDF visuals fully cover
        the collision proxies and we want to avoid z-fighting.

        The "ground" instance is always preserved so the floor stays visible,
        and the URDF visual meshes (registered as ``visual_*`` instances) are
        preserved too — they are exactly what we want to keep showing when the
        collision proxies are hidden. (This method may run after the visuals are
        registered, so it must not scale them away.)
        """
        renderer = self._sim_renderer
        for name, (instance_id, body, shape, transform, scale, c1, c2) in list(
            renderer._instances.items()
        ):
            if name == "ground" or name.startswith("terrain") or name.startswith("visual_"):
                continue
            renderer._instances[name] = (
                instance_id,
                body,
                shape,
                transform,
                (1e-6, 1e-6, 1e-6),
                c1,
                c2,
            )
        renderer._add_shape_instances = True

    def _recolor_ground(self, color1: tuple, color2: tuple) -> None:
        """Override the ground plane's checker colors.

        ``render_ground`` bakes in light grey colors that wash out against the
        Newton-style darker sky. This swaps them for darker neutral tiles.
        """
        renderer = self._sim_renderer
        if "ground" not in renderer._instances:
            return
        instance_id, body, shape, transform, scale, _c1, _c2 = renderer._instances["ground"]
        renderer._instances["ground"] = (
            instance_id,
            body,
            shape,
            transform,
            scale,
            color1,
            color2,
        )
        if shape < len(renderer._shapes):
            verts, idx_arr, _c1, _c2, geo_hash = renderer._shapes[shape]
            renderer._shapes[shape] = (verts, idx_arr, color1, color2, geo_hash)
        renderer._add_shape_instances = True

    def _patch_sky_haze(self, haze_color: tuple) -> None:
        """Replace the hard-coded orange haze color in the sky shader.

        The base renderer sets ``color2`` (haze) to ``(0.8, 0.4, 0.05)`` inside
        ``__init__``; we override it post-init so the horizon reads as a neutral
        Newton-like gradient instead of a sunset.
        """
        from pyglet import gl

        renderer = self._sim_renderer
        loc = getattr(renderer, "_loc_sky_color2", None)
        if loc is None:
            return
        with renderer._sky_shader:
            gl.glUniform3f(loc, *haze_color)

    def _patch_sun_direction(self, direction: tuple) -> None:
        """Override the shape shader's sun direction with a more raking angle.

        The base renderer sets sun to roughly straight overhead, which makes
        terrain bumps almost invisible (cos(angle) ≈ 1 everywhere). A lower
        sun lengthens the cos-shading swing across small tilts, which is the
        only mechanism the forward-shaded pipeline has to communicate surface
        roughness (no shadow maps).
        """
        from pyglet import gl

        renderer = self._sim_renderer
        # Normalise so the diffuse term in the shader stays in [-1, 1].
        d = np.asarray(direction, dtype=np.float32)
        n = float(np.linalg.norm(d))
        if n <= 1e-9:
            return
        d = d / n
        # Keep the renderer's cached vector in sync — used by the sky shader's
        # sun-glow term and any subsequent shader rebuilds.
        renderer._sun_direction = d
        loc = gl.glGetUniformLocation(
            renderer._shape_shader.id, b"sunDirection"
        )
        if loc == -1:
            return
        with renderer._shape_shader:
            gl.glUniform3f(loc, float(d[0]), float(d[1]), float(d[2]))

    # ------------------------------------------------------------------
    # window helpers
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._sim_renderer.is_running()

    def save(self) -> None:
        # No-op for the live OpenGL backend so per-frame save() calls don't block.
        pass

    def wait_until_closed(self) -> None:
        """Block until the window is closed (useful after a rollout finishes)."""
        self._sim_renderer.save()
