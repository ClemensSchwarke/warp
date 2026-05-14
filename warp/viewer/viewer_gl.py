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
_TERRAIN_COLOR = (0.50, 0.50, 0.50)    # light grey rough terrain (albedo;
                                        # the shader stacks ambient + sun
                                        # diffuse + a second diffuse + specular,
                                        # so anything > ~0.55 saturates to white)
_CONTACT_COLOR = (1.00, 0.40, 0.00)    # orange
_NORMAL_COLOR = (1.00, 0.95, 0.20)     # bright yellow

# Sky / ground colors (modeled on newton viewer's neutral palette).
_SKY_COLOR = (0.22, 0.24, 0.28)        # dark blue-grey background
_SKY_HAZE = (0.40, 0.42, 0.48)         # lighter neutral haze near the horizon
_GROUND_TILE_A = (0.38, 0.40, 0.43)    # darker tile
_GROUND_TILE_B = (0.30, 0.32, 0.34)    # secondary tile


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
        show_joints: bool = False,
        screen_width: int = 1280,
        screen_height: int = 720,
        scaling: float = 1.0,
        contact_points_radius: float = 0.02,
        urdf_path: Optional[str] = None,
        urdf_scale: float = 1.0,
        hide_collision_shapes: bool = False,
    ):
        super().__init__()
        self.model = model
        self.show_contacts = show_contacts
        self._contact_points_radius = contact_points_radius

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
            # When show_contacts=True, the underlying SimRenderer.render() will
            # transform model.rigid_contact_point0/1 into world coordinates
            # every frame and draw them as orange/blue points. Users can still
            # call viewer.log_contacts(...) for custom contact sets.
            show_rigid_contact_points=show_contacts,
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

        # Optionally hide collision shapes by scaling them to near-zero so the
        # visual meshes don't fight with them for pixels. Disabled by default
        # to keep behaviour explicit.
        if hide_collision_shapes:
            self._hide_collision_shapes()

        # Re-color the ground plane with Newton-like darker tiles (the base
        # renderer's render_ground bakes in light grey 200/150 tones).
        self._recolor_ground(_GROUND_TILE_A, _GROUND_TILE_B)

        # Load and register URDF visual meshes.
        if urdf_path is not None:
            self._register_urdf_visuals(urdf_path, urdf_scale)

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
    def _register_urdf_visuals(self, urdf_path: str, urdf_scale: float) -> None:
        body_names = list(getattr(self.model, "body_name", []))
        if not body_names:
            return
        # The body_name list on `model` repeats across envs (one set of names
        # per env). `SimRenderer.populate` namespaces body names with index
        # prefixes so they collide-free across envs. To attach visuals only to
        # the first env (visualisation purposes), we use the first
        # `bodies_per_env` entries.
        bodies_per_env = max(1, self.model.body_count // max(1, self.model.num_envs))
        first_env_body_names = body_names[:bodies_per_env]

        entries = load_urdf_visuals(urdf_path, first_env_body_names, scale=urdf_scale)
        if not entries:
            return

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

        The "ground" instance is always preserved so the floor stays visible.
        """
        renderer = self._sim_renderer
        for name, (instance_id, body, shape, transform, scale, c1, c2) in list(
            renderer._instances.items()
        ):
            if name == "ground" or name.startswith("terrain"):
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
