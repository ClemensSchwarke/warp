"""USD viewer backend for warp 1.0.0-beta.2.

Thin wrapper around ``wp.sim.render.SimRendererUsd`` that exposes the same
``ViewerBase`` API as ``ViewerGL``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from .viewer import ViewerBase


class ViewerUSD(ViewerBase):
    def __init__(
        self,
        model: warp.sim.Model,
        stage_path: str,
        fps: int = 60,
        up_axis: str = "Y",
        scaling: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self._sim_renderer = warp.sim.render.SimRendererUsd(
            model=model,
            path=stage_path,
            scaling=scaling,
            fps=fps,
            up_axis=up_axis,
        )

    def begin_frame(self, time: float) -> None:
        self.time = time
        self._sim_renderer.begin_frame(time)

    def end_frame(self) -> None:
        self._sim_renderer.end_frame()

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
        )
        self._terrain_logged = True

    def _render_contact_points(self, positions: np.ndarray, radius: float) -> None:
        self._sim_renderer.render_points(
            name="contacts",
            points=positions,
            radius=radius,
            colors=[(1.0, 0.4, 0.0)] * len(positions),
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
            color=(1.0, 1.0, 0.0),
            radius=0.005,
        )

    def save(self) -> None:
        self._sim_renderer.save()
