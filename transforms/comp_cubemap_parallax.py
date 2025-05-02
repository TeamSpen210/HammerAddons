"""Adds keys to generated cubemap materials to map them to the bounds of a cubeoid."""
import attrs

from srctools import Matrix, Vec, conv_float
from srctools.vmt import Material
from srctools.logger import get_logger

from hammeraddons.bsp_transform import trans, Context

import io
import re


LOGGER = get_logger(__name__)
# 4x4 matrix, including translation.
type Matrix4 = tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
]


def matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    """Multiply two 4x4 matrixes."""
    def helper(a: Matrix4, b: Matrix4, x: int, y: int) -> float:
        """Calculate a single result value."""
        return a[x] * b[y * 4] + a[x + 4] * b[y * 4 + 1] + a[x + 8] * b[y * 4 + 2] + a[x + 12] * b[y * 4 + 3]

    return (
        helper(a, b, 0, 0), helper(a, b, 1, 0), helper(a, b, 2, 0), helper(a, b, 3, 0),
        helper(a, b, 0, 1), helper(a, b, 1, 1), helper(a, b, 2, 1), helper(a, b, 3, 1),
        helper(a, b, 0, 2), helper(a, b, 1, 2), helper(a, b, 2, 2), helper(a, b, 3, 2),
        helper(a, b, 0, 3), helper(a, b, 1, 3), helper(a, b, 2, 3), helper(a, b, 3, 3),
    )


@attrs.define
class Config:
    """Configuration of a parallax entity."""
    origin: Vec
    radius: float
    radius_sqr: float
    used: int
    obb1: str
    obb2: str
    obb3: str


@trans('comp_cubemap_parallax')
def comp_cubemap_parallax(ctx: Context) -> None:
    """Modify cubemap materials to contain parallax information."""
    parallax_cubemap_configs: list[Config] = []
    for parallax in ctx.vmf.by_class['comp_cubemap_parallax']:
        parallax.remove()

        origin = Vec.from_str(parallax['origin'])
        angles = Matrix.from_angstr(parallax['angles'])
        radius = conv_float(parallax['radius'])
        mins, maxs = Vec.bbox(
            Vec.from_str(parallax['mins']),
            Vec.from_str(parallax['maxs'])
        )
        size = maxs - mins

        # ensure bounding box has volume
        size.max((1.0, 1.0, 1.0))

        # We need a 4-component matrix here because we need to translate
        translate1_matrix = (
            1.0, 0.0, 0.0, -origin[0],
            0.0, 1.0, 0.0, -origin[1],
            0.0, 0.0, 1.0, -origin[2],
            0.0, 0.0, 0.0, 1.0,
        )

        rotation_matrix = matmul(translate1_matrix, (
            angles[0, 0], angles[0, 1], angles[0, 2], 0.0,
            angles[1, 0], angles[1, 1], angles[1, 2], 0.0,
            angles[2, 0], angles[2, 1], angles[2, 2], 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

        translate2_matrix = matmul(rotation_matrix, (
            1.0, 0.0, 0.0, -mins.x,
            0.0, 1.0, 0.0, -mins.y,
            0.0, 0.0, 1.0, -mins.z,
            0.0, 0.0, 0.0, 1.0,
        ))

        scale_matrix = matmul(translate2_matrix, (
            1.0 / size.x, 0.0, 0.0, 0.0,
            0.0, 1.0 / size.y, 0.0, 0.0,
            0.0, 0.0, 1.0 / size.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ))

        parallax_cubemap_configs.append(Config(
            origin=origin,
            radius=radius,
            radius_sqr=radius**2,
            used=0,
            obb1=f"[{scale_matrix[0]:f} {scale_matrix[1]:f} {scale_matrix[2]:f} {scale_matrix[3]:f}]",
            obb2=f"[{scale_matrix[4]:f} {scale_matrix[5]:f} {scale_matrix[6]:f} {scale_matrix[7]:f}]",
            obb3=f"[{scale_matrix[8]:f} {scale_matrix[9]:f} {scale_matrix[10]:f} {scale_matrix[11]:f}]",
        ))

    cubemap_material_name_pattern = re.compile(r"materials/maps/.*_(-?[0-9]+)_(-?[0-9]+)_(-?[0-9]+)\.vmt")
    for name in ctx.bsp.pakfile.namelist():
        match = cubemap_material_name_pattern.fullmatch(name)
        if match is None:
            continue

        cubemap_origin = Vec(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        best_match: Config | None = None
        best_match_distance_sqr = -1.0
        for config in parallax_cubemap_configs:
            distance_sqr = (cubemap_origin - config.origin).len_sq()
            if distance_sqr > config.radius_sqr:
                continue
            if best_match is None or best_match_distance_sqr > distance_sqr:
                best_match = config
                best_match_distance_sqr = distance_sqr

        if best_match is None:
            continue

        try:
            material = Material.parse(ctx.bsp.pakfile.read(name).decode('utf-8'), filename=name)
        except Exception as exc:
            LOGGER.exception(
                "Could not parse packed cubemap patch material {}!", 
                name, 
                exc_info=exc,
            )
            continue

        if material.shader != 'patch':
            LOGGER.error(
                'Expected cubemap material to have the "patch" shader, but shader is "{}" for {}',
                material.shader,
                name
            )
            continue

        if len(material.blocks) != 1 or (material.blocks[0].name != 'replace' and material.blocks[0].name != 'insert'):
            LOGGER.error(
                'Expected cubemap material to have exactly one block named either "replace" or "insert" for {}',
                name
            )
            continue

        best_match.used += 1

        material.blocks[0].name = 'insert'
        with material.blocks[0].build() as builder:
            builder['$envmapparallax']('1')
            builder['$envmapparallaxobb1'](best_match.obb1)
            builder['$envmapparallaxobb2'](best_match.obb2)
            builder['$envmapparallaxobb3'](best_match.obb3)
            builder['$envmaporigin'](f'[{cubemap_origin}]')

        encoded = io.StringIO()
        material.export(encoded)
        ctx.pack.pack_file(name, data=encoded.getvalue().encode('utf8'))

    for config in parallax_cubemap_configs:
        if config.used == 0:
            LOGGER.warning(
                'No materials found affected by a cubemap within {} units for comp_cubemap_parallax at ({})!',
                config.radius,
                config.origin,
            )
