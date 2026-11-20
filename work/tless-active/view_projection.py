"""Predict which camera rays can distinguish two metric CAD hypotheses."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, minimum_filter

from cad_consensus import transform


@dataclass(frozen=True)
class Projection:
    first: np.ndarray
    second: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    visibility: np.ndarray
    selected: np.ndarray
    priority: float


def intrinsics(camera):
    fx, _, cx, _, fy, cy, *_ = camera["K"]
    return np.array([fx, fy]), np.array([cx, cy])


def point_cloud(depth, camera):
    """Back-project the first measured depth image into world coordinates."""
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[rows, cols].reshape(-1, 1)
    focal, principal = intrinsics(camera)
    local = np.column_stack(((np.column_stack((cols, rows)) - principal) * z / focal, z))
    rotation, translation = transform(camera)
    return np.asarray((local - translation) @ rotation, dtype=np.float32)


def project_vertices(vertices, pose, rotation, translation, focal, principal):
    model_rotation, model_translation = pose
    local = vertices @ (rotation @ model_rotation).T + rotation @ model_translation + translation
    z = local[:, 2]
    if np.min(z) <= 0 or not np.all(np.isfinite(local)):
        return None
    image = local[:, :2] * focal / z[:, None] + principal
    # Inverse depth interpolates linearly after a perspective projection.
    return np.column_stack((image, -1. / z))


def raster_depth(vertices, faces, lower, width, height, render):
    size = max(width, height)
    vertices[:, :2] -= lower + (size - 1) / 2
    buffer = np.full((size, size), np.inf, dtype=np.float32)
    render(np.ascontiguousarray(vertices, dtype=np.float32), faces, len(faces), size, 1., buffer)
    buffer = buffer[:height, :width]
    result = np.full(buffer.shape, np.inf, dtype=np.float32)
    np.divide(-1., buffer, out=result, where=np.isfinite(buffer))
    return result


def known_depth(cloud, rotation, translation, focal, principal, lower, width, height):
    local = cloud @ rotation.T + translation
    local = local[np.isfinite(local).all(axis=1) & (local[:, 2] > 0)]
    z = local[:, 2]
    xy = np.rint(local[:, :2] * focal / z[:, None] + principal - lower).astype(int)
    cols, rows = xy.T
    in_crop = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    nearest = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(nearest, (rows[in_crop], cols[in_crop]), z[in_crop])
    # A one-pixel margin covers gaps left by reprojecting isolated samples.
    return minimum_filter(nearest, size=3, mode="constant", cval=np.inf)


def project_pair(first_mesh, first_pose, second_mesh, second_pose, camera, points, render, shape,
                 *, depth_gap=8., occlusion_gap=12.):
    """Rank a camera using CAD predictions and only previously measured points.

    The returned arrays share a clipped crop. Infinite model depth denotes a
    silhouette absence. Camera poses and measured depths are in millimeters.
    """
    rotation, translation = transform(camera)
    focal, principal = intrinsics(camera)
    first_verts = project_vertices(first_mesh[0], first_pose, rotation, translation, focal, principal)
    second_verts = project_vertices(second_mesh[0], second_pose, rotation, translation, focal, principal)
    if first_verts is None or second_verts is None:
        return None

    low = np.minimum(first_verts[:, :2].min(axis=0), second_verts[:, :2].min(axis=0))
    high = np.maximum(first_verts[:, :2].max(axis=0), second_verts[:, :2].max(axis=0))
    lower = np.maximum(np.floor(low), 0).astype(int)
    upper = np.minimum(np.ceil(high), np.asarray(shape[::-1]) - 1).astype(int)
    width, height = upper - lower + 1
    if min(width, height) <= 0:
        return None

    first = raster_depth(first_verts, first_mesh[1], lower, width, height, render)
    second = raster_depth(second_verts, second_mesh[1], lower, width, height, render)
    exists_first = np.isfinite(first)
    exists_second = np.isfinite(second)
    union = exists_first | exists_second
    if not union.any():
        return None

    known = known_depth(points, rotation, translation, focal, principal, lower, width, height)
    visibility = union & (known + occlusion_gap >= np.minimum(first, second))
    core_first = binary_erosion(exists_first)
    core_second = binary_erosion(exists_second)
    common = core_first & core_second & visibility
    difference = np.zeros(first.shape, dtype=np.float32)
    np.subtract(first, second, out=difference, where=common)
    different_depth = common & (np.abs(difference) >= depth_gap)
    # Leave one pixel of separation around the silhouette of the rival.
    silhouette = visibility & ((core_first & ~binary_dilation(exists_second))
                               | (core_second & ~binary_dilation(exists_first)))
    selected = different_depth | silhouette
    depth_weight = np.minimum(np.abs(difference[different_depth]) / (2 * depth_gap), 1.)
    priority = (depth_weight.sum() + .75 * np.count_nonzero(silhouette)) / np.sqrt(np.count_nonzero(union))
    rows = np.arange(lower[1], upper[1] + 1)
    cols = np.arange(lower[0], upper[0] + 1)
    return Projection(first, second, rows, cols, visibility, selected, float(priority))
