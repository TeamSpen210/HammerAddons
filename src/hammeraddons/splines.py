"""Math related to spline generation."""
import math
from typing import Generator

from srctools import Matrix, Vec


def parallel_transport(orient1: Matrix, tanj1: Vec, tanj2: Vec) -> Matrix:
    """Given a series of vectors along a path, produce a corresponding set of orientations.

    This allows creating geometry which smoothly follows a curve.
    The code is based on the info at: https://janakiev.com/blog/framing-parametric-curves/.
    """
    b = Vec.cross(tanj1, tanj2)
    if b.mag_sq() < 0.001:
        # Aligned, yield the same orientation.
        return orient1.copy()
    else:
        phi = math.acos(Vec.dot(tanj1, tanj2))
        up = orient1.up() @ Matrix.axis_angle(b.norm(), math.degrees(phi))
        return Matrix.from_basis(x=tanj2, z=up)
