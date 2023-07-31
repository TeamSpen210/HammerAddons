"""Adds keys to generated cubemap materials to map them to the bounds of a cubeoid."""

from srctools import Matrix, Vec, conv_float
from srctools.logger import get_logger

from hammeraddons.bsp_transform import trans, Context

import re


LOGGER = get_logger(__name__)


@trans('comp_cubemap_parallax')
def comp_cubemap_parallax(ctx: Context):
    """Modify cubemap materials to contain parallax information."""
    cubemap_material_name_pattern = re.compile(r"materials/maps/.*_(-?[0-9]+)_(-?[0-9]+)_(-?[0-9]+)\.vmt")
    cubemap_material_extract_pattern = re.compile(br'"patch"\r\n\{\r\n\t"include"\t\t"([^"]*)"\r\n\t"(?:replace|insert)"\r\n\t\{\r\n\t\t"\$envmap"\t\t"([^"]*)"\r\n(.*)\t\}\r\n}\r\n')
    for parallax in ctx.vmf.by_class['comp_cubemap_parallax']:
        parallax.remove()

        origin = Vec.from_str(parallax['origin'])
        angles = Matrix.from_angstr(parallax['angles'])
        radius = conv_float(parallax['radius'])
        radius_squared = radius**2
        mins = Vec.from_str(parallax['mins'])
        maxs = Vec.from_str(parallax['maxs'])
        diff = maxs - mins

        # ensure bounding box has volume
        if diff[0] == 0.0:
            diff[0] = 1.0
        if diff[1] == 0.0:
            diff[1] = 1.0
        if diff[2] == 0.0:
            diff[2] = 1.0

        # we need a 4-component matrix here because we need to translate
        def matmul(a, b):
            def helper(a, b, x, y):
                return a[x] * b[y * 4] + a[x + 4] * b[y * 4 + 1] + a[x + 8] * b[y * 4 + 2] + a[x + 12] * b[y * 4 + 3]

            return (
                helper(a, b, 0, 0), helper(a, b, 1, 0), helper(a, b, 2, 0), helper(a, b, 3, 0),
                helper(a, b, 0, 1), helper(a, b, 1, 1), helper(a, b, 2, 1), helper(a, b, 3, 1),
                helper(a, b, 0, 2), helper(a, b, 1, 2), helper(a, b, 2, 2), helper(a, b, 3, 2),
                helper(a, b, 0, 3), helper(a, b, 1, 3), helper(a, b, 2, 3), helper(a, b, 3, 3),
            )

        translate1_matrix = (
            1.0, 0.0, 0.0, -origin[0],
            0.0, 1.0, 0.0, -origin[1],
            0.0, 0.0, 1.0, -origin[2],
            0.0, 0.0, 0.0, 1.0,
        )

        rotation_matrix = matmul(translate1_matrix, (
            angles[(0, 0)], angles[(0, 1)], angles[(0, 2)], 0.0,
            angles[(1, 0)], angles[(1, 1)], angles[(1, 2)], 0.0,
            angles[(2, 0)], angles[(2, 1)], angles[(2, 2)], 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

        translate2_matrix = matmul(rotation_matrix, (
            1.0, 0.0, 0.0, -mins[0],
            0.0, 1.0, 0.0, -mins[1],
            0.0, 0.0, 1.0, -mins[2],
            0.0, 0.0, 0.0, 1.0,
        ))

        scale_matrix = matmul(translate2_matrix, (
            1.0 / diff[0], 0.0, 0.0, 0.0,
            0.0, 1.0 / diff[1], 0.0, 0.0,
            0.0, 0.0, 1.0 / diff[2], 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

        parallax_material_keys = (b"\t\t\"$envmapparallax\"\t\t\"1\"\r\n" +
            b"\t\t\"$envmapparallaxobb1\"\t\t\"[%f %f %f %f]\"\r\n" +
            b"\t\t\"$envmapparallaxobb2\"\t\t\"[%f %f %f %f]\"\r\n" +
            b"\t\t\"$envmapparallaxobb3\"\t\t\"[%f %f %f %f]\"\r\n") % (
                scale_matrix[0], scale_matrix[1], scale_matrix[2], scale_matrix[3],
                scale_matrix[4], scale_matrix[5], scale_matrix[6], scale_matrix[7],
                scale_matrix[8], scale_matrix[9], scale_matrix[10], scale_matrix[11],
            )

        any_found = False
        for name in ctx.bsp.pakfile.namelist():
            match = cubemap_material_name_pattern.fullmatch(name)
            if match is None:
                continue

            cubemap_origin = Vec(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if (cubemap_origin - origin).len_sq() > radius_squared:
                continue

            any_found = True
            material_bytes = ctx.bsp.pakfile.read(name)
            match = cubemap_material_extract_pattern.fullmatch(material_bytes)
            if match is None:
                LOGGER.error(
                    'Failed to parse cubemap material {}',
                    name
                )
                continue

            ctx.pack.pack_file(name, data = (b"\"patch\"\r\n{\r\n\t\"include\"\t\t\"" + match.group(1) +
                b"\"\r\n\t\"insert\"\r\n\t{\r\n\t\t\"$envmap\"\t\t\"" + match.group(2) +
                b"\"\r\n" + match.group(3) + parallax_material_keys +
                b"\t\t\"$envmaporigin\"\t\t\"[" + bytes(str(cubemap_origin), 'ascii') +
                b"]\"\r\n\t}\r\n}\r\n"))

        if not any_found:
            LOGGER.warning(
                'No cubemapped materials found within {} units for comp_cubemap_parallax at ({})!',
                parallax['radius'],
                parallax['origin'],
            )
