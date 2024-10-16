from typing import Any

import warp as wp

from warp.fem.types import ElementIndex, NULL_ELEMENT_INDEX
from warp.fem.utils import masked_indices

from .geometry import Geometry


wp.set_module_options({"enable_backward": False})


class GeometryPartition:

    """Base class for geometry partitions, i.e. subset of cells and sides"""

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def cell_count(self) -> int:
        """Number of cells that are 'owned' by this partition"""
        raise NotImplementedError()

    def side_count(self) -> int:
        """Number of sides that are 'owned' by this partition"""
        raise NotImplementedError()

    def boundary_side_count(self) -> int:
        """Number of geo-boundary sides that are 'owned' by this partition"""
        raise NotImplementedError()

    def frontier_side_count(self) -> int:
        """Number of sides with neighbors owned by this and another partition"""
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return f"{self.geometry.name}_{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.name


class WholeGeometryPartition(GeometryPartition):
    """Trivial (NOP) partition"""

    def __init__(
        self,
        geometry: Geometry,
    ):
        super().__init__(geometry)

        self.SideArg = geometry.SideIndexArg
        self.side_arg_value = geometry.side_index_arg_value

        self.cell_index = WholeGeometryPartition._identity_element_index
        self.partition_cell_index = WholeGeometryPartition._identity_element_index

        self.side_index = WholeGeometryPartition._identity_element_index
        self.boundary_side_index = geometry.boundary_side_index
        self.frontier_side_index = WholeGeometryPartition._identity_element_index

    def __eq__(self, other: GeometryPartition) -> bool:
        # Ensures that two whole partition instances of the same geometry are considered equal
        return isinstance(other, WholeGeometryPartition) and self.geometry == other.geometry

    def cell_count(self) -> int:
        return self.geometry.cell_count()

    def side_count(self) -> int:
        return self.geometry.side_count()

    def boundary_side_count(self) -> int:
        return self.geometry.boundary_side_count()

    def frontier_side_count(self) -> int:
        return 0

    @wp.struct
    class CellArg:
        pass

    def cell_arg_value(self, device):
        arg = WholeGeometryPartition.CellArg()
        return arg

    @wp.func
    def _identity_element_index(args: Any, idx: ElementIndex):
        return idx


class CellBasedGeometryPartition(GeometryPartition):
    """Geometry partition based on a subset of cells. Interior, boundary and frontier sides are automatically categorized."""

    def __init__(
        self,
        geometry: Geometry,
        device=None,
    ):
        super().__init__(geometry)

    @wp.struct
    class SideArg:
        partition_side_indices: wp.array(dtype=int)
        boundary_side_indices: wp.array(dtype=int)
        frontier_side_indices: wp.array(dtype=int)

    def side_count(self) -> int:
        return self._partition_side_indices.shape[0]

    def boundary_side_count(self) -> int:
        return self._boundary_side_indices.shape[0]

    def frontier_side_count(self) -> int:
        return self._frontier_side_indices.shape[0]

    def side_arg_value(self, device):
        arg = LinearGeometryPartition.SideArg()
        arg.partition_side_indices = self._partition_side_indices.to(device)
        arg.boundary_side_indices = self._boundary_side_indices.to(device)
        arg.frontier_side_indices = self._frontier_side_indices.to(device)
        return arg

    @wp.func
    def side_index(args: SideArg, partition_side_index: int):
        """partition side to side index"""
        return args.partition_side_indices[partition_side_index]

    @wp.func
    def boundary_side_index(args: SideArg, boundary_side_index: int):
        """Boundary side to side index"""
        return args.boundary_side_indices[boundary_side_index]

    @wp.func
    def frontier_side_index(args: SideArg, frontier_side_index: int):
        """Frontier side to side index"""
        return args.frontier_side_indices[frontier_side_index]

    def compute_side_indices_from_cells(
        self,
        cell_arg_value: Any,
        cell_inclusion_test_func: wp.Function,
        device,
    ):
        from warp.fem import cache

        def count_side_fn(
            geo_arg: self.geometry.SideArg,
            cell_arg_value: Any,
            partition_side_mask: wp.array(dtype=int),
            boundary_side_mask: wp.array(dtype=int),
            frontier_side_mask: wp.array(dtype=int),
        ):
            side_index = wp.tid()
            inner_cell_index = self.geometry.side_inner_cell_index(geo_arg, side_index)
            outer_cell_index = self.geometry.side_outer_cell_index(geo_arg, side_index)

            inner_in = cell_inclusion_test_func(cell_arg_value, inner_cell_index)
            outer_in = cell_inclusion_test_func(cell_arg_value, outer_cell_index)

            if inner_in:
                # Inner neighbor in partition; count as partition side
                partition_side_mask[side_index] = 1

                # Inner and outer element as the same -- this is a boundary side
                if inner_cell_index == outer_cell_index:
                    boundary_side_mask[side_index] = 1

            if inner_in != outer_in:
                # Exactly one neighbor in partition; count as frontier side
                frontier_side_mask[side_index] = 1

        count_sides = cache.get_kernel(
            count_side_fn,
            suffix=f"{self.geometry.name}_{cell_inclusion_test_func.key}",
        )

        partition_side_mask = wp.zeros(
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        boundary_side_mask = wp.zeros(
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )
        frontier_side_mask = wp.zeros(
            shape=(self.geometry.side_count(),),
            dtype=int,
            device=device,
        )

        wp.launch(
            dim=partition_side_mask.shape[0],
            kernel=count_sides,
            inputs=[
                self.geometry.side_arg_value(device),
                cell_arg_value,
                partition_side_mask,
                boundary_side_mask,
                frontier_side_mask,
            ],
            device=device,
        )

        # Convert counts to indices
        self._partition_side_indices, _ = masked_indices(partition_side_mask)
        self._boundary_side_indices, _ = masked_indices(boundary_side_mask)
        self._frontier_side_indices, _ = masked_indices(frontier_side_mask)


class LinearGeometryPartition(CellBasedGeometryPartition):
    def __init__(
        self,
        geometry: Geometry,
        partition_rank: int,
        partition_count: int,
        device=None,
    ):
        """Creates a geometry partition by uniformly partionning cell indices

        Args:
            geometry: the geometry to partition
            partition_rank: the index of the partition being created
            partition_count: the number of partitions that will be created over the geometry
            device: Warp device on which to perform and store computations
        """
        super().__init__(geometry)

        total_cell_count = geometry.cell_count()

        cells_per_partition = (total_cell_count + partition_count - 1) // partition_count
        self.cell_begin = cells_per_partition * partition_rank
        self.cell_end = min(self.cell_begin + cells_per_partition, total_cell_count)

        super().compute_side_indices_from_cells(
            self.cell_arg_value(device),
            LinearGeometryPartition._cell_inclusion_test,
            device,
        )

    def cell_count(self) -> int:
        return self.cell_end - self.cell_begin

    @wp.struct
    class CellArg:
        cell_begin: int
        cell_end: int

    def cell_arg_value(self, device):
        arg = LinearGeometryPartition.CellArg()
        arg.cell_begin = self.cell_begin
        arg.cell_end = self.cell_end
        return arg

    @wp.func
    def cell_index(args: CellArg, partition_cell_index: int):
        """Partition cell to cell index"""
        return args.cell_begin + partition_cell_index

    @wp.func
    def partition_cell_index(args: CellArg, cell_index: int):
        """Partition cell to cell index"""
        if cell_index > args.cell_end:
            return NULL_ELEMENT_INDEX

        partition_cell_index = cell_index - args.cell_begin
        if partition_cell_index < 0:
            return NULL_ELEMENT_INDEX

        return partition_cell_index

    @wp.func
    def _cell_inclusion_test(arg: CellArg, cell_index: int):
        return cell_index >= arg.cell_begin and cell_index < arg.cell_end


class ExplicitGeometryPartition(CellBasedGeometryPartition):
    def __init__(self, geometry: Geometry, cell_mask: "wp.array(dtype=int)"):
        """Creates a geometry partition by uniformly partionning cell indices

        Args:
            geometry: the geometry to partition
            cell_mask: warp array of length ``geometry.cell_count()`` indicating which cells are selected. Array values must be either ``1`` (selected) or ``0`` (not selected).
        """

        super().__init__(geometry)

        self._cell_mask = cell_mask
        self._cells, self._partition_cells = masked_indices(self._cell_mask)

        super().compute_side_indices_from_cells(
            self._cell_mask,
            ExplicitGeometryPartition._cell_inclusion_test,
            self._cell_mask.device,
        )

    def cell_count(self) -> int:
        return self._cells.shape[0]

    @wp.struct
    class CellArg:
        cell_index: wp.array(dtype=int)
        partition_cell_index: wp.array(dtype=int)

    def cell_arg_value(self, device):
        arg = ExplicitGeometryPartition.CellArg()
        arg.cell_index = self._cells.to(device)
        arg.partition_cell_index = self._partition_cells.to(device)
        return arg

    @wp.func
    def cell_index(args: CellArg, partition_cell_index: int):
        return args.cell_index[partition_cell_index]

    @wp.func
    def partition_cell_index(args: CellArg, cell_index: int):
        return args.partition_cell_index[cell_index]

    @wp.func
    def _cell_inclusion_test(mask: wp.array(dtype=int), cell_index: int):
        return mask[cell_index] > 0
