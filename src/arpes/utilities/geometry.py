"""Geometry and intersection utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    "polyhedron_intersect_plane",
    "segment_contains_point",
    "point_plane_intersection",
)


def point_plane_intersection(
    plane_normal: NDArray[np.float_],
    plane_point: NDArray[np.float_],
    line_a: NDArray[np.float_],
    line_b: NDArray[np.float_],
    epsilon: float = 1e-6,
) -> NDArray[np.float_] | None:
    """Determines the point plane intersection.

    The plane is defined by a point and a normal vector while the line is defined by line_a
    and line_b. All should be numpy arrays.

    Args:
        plane_normal: The normal vector of the plane.
        plane_point: The point in the plane.
        line_a: The line A.
        line_b: The line B.
        epsilon: Precision of the line difference.

    Returns:
        The intersection point of the point and plane.
    """
    line_direction = line_b - line_a
    if abs(plane_normal.dot(line_direction)) < epsilon:
        return None

    delta = line_a - plane_point
    projection = -plane_normal.dot(delta) / plane_normal.dot(line_direction)

    return delta + projection * line_direction + plane_point


def segment_contains_point(
    line_a: NDArray[np.float_],
    line_b: NDArray[np.float_],
    point_along_line: NDArray[np.float_] | None,
    epsilon: float = 1e-6,
    *,
    check: bool = False,
) -> bool:
    """Determines whether a segment contains a point that also lies along the line.

    If asked to check, it will also return false if the point does not lie along the line.
    """
    if point_along_line is None:
        return False

    delta = line_b - line_a
    delta_p = point_along_line - line_a
    if check:
        cosine = delta.dot(delta_p) / (np.linalg.norm(delta) * np.linalg.norm(delta_p))
        if cosine < 1 - epsilon:
            return False

    return 0 - epsilon < delta.dot(delta_p) / delta.dot(delta) < 1 + epsilon


def polyhedron_intersect_plane(
    poly_faces: list[NDArray[np.float_]],
    plane_normal: NDArray[np.float_],
    plane_point: NDArray[np.float_],
    epsilon: float = 1e-6,
) -> ConvexHull:
    """Determines the intersection of a convex polyhedron intersecting a plane.

    The polyhedron faces should be given by a list of np.arrays, where each np.array at
    index `i` is the vertices of face `i`.

    As an example, running [p[0] for p in ase.dft.bz.bz_vertices(np.linalg.inv(cell).T)]
    should provide acceptable input for a unit cell `cell`.

    The polyhedron should be convex because we construct the convex hull in order to order
    the points.

    Args:
        poly_faces: The faces of the polyhedron as a list of arrays with with the polygonal
          facial vertices
        plane_normal: Normal vector to the plan
        plane_point: Any point on the plane
        epsilon: Used to determine precision for non-intersection
    """
    collected_points = []

    def add_point(c: NDArray[np.float_]) -> None:
        already_collected = False
        for other in collected_points:
            delta = c - other
            if delta.dot(delta) < epsilon:
                already_collected = True
                break

        if not already_collected:
            collected_points.append(c)

    for poly_face in poly_faces:
        segments = list(
            zip(poly_face, np.concatenate([poly_face[1:], [poly_face[0]]]), strict=True),
        )
        for a, b in segments:
            intersection = point_plane_intersection(
                plane_normal,
                plane_point,
                a,
                b,
                epsilon=epsilon,
            )
            if segment_contains_point(a, b, intersection, epsilon=epsilon):
                add_point(intersection)

    points = ConvexHull(collected_points, qhull_options="Qc QJ").points

    # sort
    for_sort = points - np.mean(points, axis=0)
    for_sort = (for_sort.T / np.linalg.norm(for_sort, axis=1)).T
    det = plane_normal.dot(np.cross(for_sort[0], for_sort).T)
    dot = for_sort[0].dot(for_sort.T)

    return points[np.argsort(np.arctan2(det, dot))]