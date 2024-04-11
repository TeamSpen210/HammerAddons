"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import math
from typing import (
    Callable, Dict, FrozenSet, Iterable, Iterator, List, Literal, MutableMapping, Optional, Set,
    Tuple,
    Union, Sequence,
)
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
import fnmatch
import itertools
import operator
import os
import re
import shutil

from srctools import (
    VMF, Entity, FileSystemChain, KeyValError, Keyvalues, bool_as_int, conv_int,
)
from srctools.bsp import BSP, BModel, StaticProp, StaticPropFlags, VisLeaf
from srctools.game import Game
from srctools.logger import get_logger
from srctools.math import Angle, Matrix, Vec, quickhull
from srctools.mdl import MDL_EXTS, Model, Flags as ModelFlags
from srctools.packlist import PackList
from srctools.smd import Bone, Mesh, Triangle, Vertex
from srctools.tokenizer import Token, Tokenizer
import attrs
import trio

from .acache import ACache
from .mdl_compiler import ModelCompiler


LOGGER = get_logger(__name__)


class CollType(Enum):
    """Collision types that static props can have."""
    NONE = 0  # No collision
    BSP = 1  # Treat the same as MODEL.
    BBOX = 2
    OBB = 3
    OBB_YAW = 4
    VPHYS = 6  # Collision model


@attrs.frozen
class QC:
    """The relevant we need from a QC."""
    path: str  # QC path.
    ref_smd: str  # Absolute location of main visible geometry.
    phy_smd: Optional[str]  # Absolute location of collision model, or None
    ref_scale: float  # Scale of main model.
    phy_scale: float  # Scale of collision model.
    is_concave: bool  # If the collision model is known to be concave.

QC_TEMPLATE = '''\
$staticprop
$modelname "{path}"
$surfaceprop "{surf}"

$body body "reference.smd"

$contents {contents}

$sequence idle anim act_idle 1
'''

QC_COLL_TEMPLATE = '''
$collisionmodel "physics.smd" {
    $maxconvexpieces 2048
    $automass
    $concave
}
'''

MAX_GROUP = 24  # StudioMDL doesn't allow more than this...
# Exceed 65k triangles and StudioMDL cuts into multiple bodygroups. So at that point we should
# produce multiple grouped props. Shrink a little just for breathing room.
MAX_VERTS = 65536//3 - 64


# Cache of the SMD models we have already parsed, so we don't need
# to parse them again. For the collision model, we store them pre-split.
_mesh_cache: ACache[Tuple[QC, int], Mesh] = ACache()
_coll_cache: ACache[Tuple[Optional[str], CollType], List[Mesh]] = ACache()

# Limit the amount of decompile/recompiles we do simultaneously.
LIM_PROCESS = trio.CapacityLimiter(8)
LIM_PARSE = trio.CapacityLimiter(16)


def unify_mdl(path: str):
    """Compute a 'canonical' path for a given model."""
    path = path.casefold().replace('\\', '/').lstrip('/')
    if not path.startswith('models/'):
        path = 'models/' + path
    if not path.endswith(('.mdl', '.glb', '.gltf')):
        path = path + '.mdl'
    return path


class CombineVolume:
    """Parsed comp_propcombine_* ents."""
    def __init__(self, group_name: str, skinset: FrozenSet, origin: Vec) -> None:
        self.group = group_name
        self.skinset = skinset
        # For sorting.
        self.volume = 0.0
        self.used = False
        self.mins: Optional[Vec] = None
        self.maxes: Optional[Vec] = None
        # Each volume in the group, specifying its collision behaviour.
        self.collision: List[Callable[[Vec], bool]] = []

        if group_name:
            self._desc_start = f'group "{group_name}"'
        else:
            self._desc_start = f'at {origin}'

    def contains(self, point: Vec) -> bool:
        """Check if the volume contains this point."""
        return any(coll(point) for coll in self.collision)

    def __str__(self) -> str:
        if self.mins is None or self.maxes is None:
            return self._desc_start
        else:
            return f'{self._desc_start}, ({self.mins} : {self.maxes})'


def make_collision_bbox(origin: Vec, angles: Angle, mins: Vec, maxes: Vec) -> Callable[[Vec], bool]:
    """Produce a bounding box collision checker."""
    # Transpose the angles, giving us the inverse transform.
    inv_angles = Matrix.from_angle(angles).transpose()

    def check(point: Vec) -> bool:
        """Check if the given position is inside the bbox."""
        local_point = (point - origin) @ inv_angles
        return local_point.in_bbox(mins, maxes)
    return check


def make_collision_brush(origin: Vec, angles: Angle, brush: BModel) -> Callable[[Vec], bool]:
    """Produce a collision checker using a brush entity."""
    # Transpose the angles, giving us the inverse transform.
    inv_angles = Matrix.from_angle(angles).transpose()

    def check(point: Vec) -> bool:
        """Check if the given position is inside the volume."""
        local_point = (point - origin) @ inv_angles
        leaf = brush.node.test_point(local_point)
        return leaf is not None and len(leaf.brushes) > 0
    return check


@attrs.frozen
class PropPos:
    """Key used to match models to each other."""
    x: float
    y: float
    z: float

    pit: float
    yaw: float
    rol: float

    model: str
    checksum: bytes
    skin: int
    
    scale_x: float
    scale_y: float
    scale_z: float

    solidity: CollType


# The types used during compilation.
PropCombiner = ModelCompiler[
    Tuple[FrozenSet[PropPos], bool],  # Key for deduplication.
    # Additional parameters used during compile
    Tuple[Callable[[str], Union[Tuple[QC, Model], Tuple[None, None]]], float],
    # Result of the function
    None,
]


async def combine_group(
    compiler: PropCombiner,
    props: List[StaticProp],
    lookup_model: Callable[[str], Union[Tuple[QC, Model], Tuple[None, None]]],
    volume_tolerance: float,
) -> StaticProp:
    """Merge the given props together, compiling a model if required."""

    # We want to allow multiple props to reuse the same model.
    # To do this try and match prop groups to each other, by "unifying"
    # them into a consistent orientation.
    #
    # If there are matches in different orientations, they're most likely
    # 90 degree or other rotations in the yaw axis. So we compute the average,
    # and subtract that out.

    avg_pos = Vec()
    avg_lighting = Vec()
    avg_yaw = 0.0

    visleafs: Set[VisLeaf] = set()

    for prop in props:
        avg_pos += prop.origin
        avg_lighting += prop.lighting
        yaw = prop.angles.yaw % 90
        if yaw > 45.0:
            avg_yaw -= 90.0 - yaw
        else:
            avg_yaw += yaw
        visleafs.update(prop.visleafs)

    avg_yaw /= len(props)
    avg_pos /= len(props)
    avg_lighting /= len(props)
    yaw_rot = Matrix.from_yaw(-avg_yaw)

    prop_pos = set()
    for prop in props:
        origin = round((prop.origin - avg_pos) @ yaw_rot, 7)
        angles = prop.angles
        angles.pitch = round(angles.pitch, 7)
        angles.yaw = round(angles.yaw - avg_yaw, 7)
        angles.roll = round(angles.roll, 7)
        try:
            coll = CollType(prop.solidity)
        except ValueError:
            raise ValueError(
                 'Unknown prop_static collision type '
                 '{} for "{}" at {}!'.format(
                    prop.solidity,
                    prop.model,
                    prop.origin,
                 )
            )
        qc, mdl = lookup_model(prop.model)
        assert mdl is not None, prop.model

        scale = prop.scaling
        if isinstance(scale, float):
            scale_x = scale_y = scale_z = scale
        else:
            scale_x, scale_y, scale_z = scale
        prop_pos.add(PropPos(
            origin.x, origin.y, origin.z,
            angles.pitch, angles.yaw, angles.roll,
            prop.model,
            mdl.checksum,
            prop.skin,
            scale_x,
            scale_y,
            scale_z,
            coll,
        ))
    # We don't want to build collisions if it's not used.
    has_coll = any(pos.solidity is not CollType.NONE for pos in prop_pos)
    mdl_name, _ = await compiler.get_model(
        (frozenset(prop_pos), has_coll),
        compile_func, (lookup_model, volume_tolerance),
    )

    # Many of these values we require to be the same, so we can read them
    # from any of the component props.
    combined = StaticProp(
        model=mdl_name,
        origin=avg_pos,
        angles=Angle(0, avg_yaw - 90, 0),
        scaling=1.0,
        visleafs=visleafs,
        solidity=(CollType.VPHYS if has_coll else CollType.NONE).value,
        flags=props[0].flags,
        lighting=avg_lighting,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
        min_fade=0.0,
        max_fade=0.0,
    )

    # Calculate a new fade distance pair that encloses the original fade distances.
    # Screen-space fade is an old method, which can't be done this way. Just don't bother.
    if StaticPropFlags.DOES_FADE in combined.flags and StaticPropFlags.SCREEN_SPACE_FADE not in combined.flags:
        for prop in props:
            distance = (prop.origin - avg_pos).mag()
            combined.min_fade = max(combined.min_fade, distance + prop.min_fade)
            combined.max_fade = max(combined.max_fade, distance + prop.max_fade)
    return combined


async def compile_func(
    mdl_key: Tuple[FrozenSet[PropPos], bool],
    temp_folder: Path,
    mdl_name: str,
    args: Tuple[Callable[[str], Union[Tuple[QC, Model], Tuple[None, None]]], float],
) -> None:
    """Build this merged model."""
    LOGGER.info('Compiling {}...', mdl_name)
    prop_pos, has_coll = mdl_key
    lookup_model, volume_tolerance = args

    # Unify these properties.
    surfprops: Set[str] = set()
    cdmats: Set[str] = set()
    contents: Set[int] = set()
    combined_flags = ModelFlags(0)

    for prop in prop_pos:
        qc, mdl = lookup_model(prop.model)
        assert qc is not None, prop.model
        assert mdl is not None, prop.model
        surfprops.add(mdl.surfaceprop.casefold())
        cdmats.update(mdl.cdmaterials)
        contents.add(mdl.contents)
        combined_flags |= mdl.flags

    if len(surfprops) > 1:
        raise ValueError('Multiple surfaceprops? Should be filtered out.')

    if len(contents) > 1:
        raise ValueError('Multiple contents? Should be filtered out.')

    [surfprop] = surfprops
    [phy_content_type] = contents

    ref_mesh = Mesh.blank('static_prop')
    coll_mesh = Mesh.blank('static_prop')
    [coll_bone] = coll_mesh.bones.values()
    bone_link = [(coll_bone, 1.0)]
    coll_groups: Dict[Mesh, float] = {}

    for prop in prop_pos:
        qc, mdl = lookup_model(prop.model)
        assert qc is not None, prop.model
        assert mdl is not None, prop.model

        child_ref = await _mesh_cache.fetch((qc, prop.skin), build_reference, prop, qc, mdl)
        child_coll = await _coll_cache.fetch((qc.phy_smd, prop.solidity), build_collision, qc, prop, child_ref, volume_tolerance > 0)

        scale = Vec(prop.scale_x, prop.scale_y, prop.scale_z)
        offset = Vec(prop.x, prop.y, prop.z)
        rot_matrix = Matrix.from_angle(prop.pit, prop.yaw, prop.rol)

        ref_mesh.append_model(child_ref, rot_matrix, offset, scale * qc.ref_scale)

        if has_coll and child_coll is not None:
            phy_scale = scale * qc.phy_scale
            
            matrix = Matrix()
            
            # Set the scale
            matrix[0, 0] = phy_scale.x
            matrix[1, 1] = phy_scale.y
            matrix[2, 2] = phy_scale.z

            # Rotate the matrix
            matrix @= rot_matrix

            # Secondary matrix for the normals
            itm = matrix.inverse().transpose()

            group = Mesh(coll_mesh.bones, coll_mesh.animation, [])
            for part in child_coll:
                for orig_tri in part.triangles:
                    new_tri = orig_tri.copy()
                    for vert in new_tri:
                        vert.links[:] = bone_link

                        # Transform the vertex
                        vert.norm @= itm
                        vert.norm = vert.norm.norm()
                        vert.pos @= matrix
                        vert.pos += offset

                    group.triangles.append(new_tri)
            if group.triangles:
                coll_groups[group] = group.compute_volume()

    with (temp_folder / 'reference.smd').open('wb') as fb:
        ref_mesh.export(fb)

    # Generate  a  blank animation.
    with (temp_folder / 'anim.smd').open('wb') as fb:
        Mesh.blank('static_prop').export(fb)

    if coll_groups:
        if volume_tolerance > 0:
            await trio.to_thread.run_sync(
                optimise_collision,
                volume_tolerance, bone_link,
                coll_mesh, coll_groups,
            )
        else:
            # Just use unaltered.
            for mesh1 in coll_groups:
                coll_mesh.triangles += mesh1.triangles
        with (temp_folder / 'physics.smd').open('wb') as fb:
            coll_mesh.export(fb)

    with (temp_folder / 'model.qc').open('w') as f:
        f.write(QC_TEMPLATE.format(
            path=mdl_name,
            surf=surfprop,
            # For $contents, we need to decompose out each bit.
            # This is the same as BSP's flags in public/bsp_flags.h
            # However only a few types are allowable.
            contents=' '.join([
                cont
                for mask, cont in [
                    (0x1, '"solid"'),
                    (0x8, '"grate"'),
                    (0x2000000, '"monster"'),
                    (0x20000000, '"ladder"'),
                ]
                if mask & phy_content_type
                # 0 needs to produce this value.
            ]) or '"notsolid"',
        ))
        # According to studiomdl, $opaque overrides $mostlyopaque.
        if ModelFlags.force_opaque in combined_flags:
            f.write('$opaque\n')
            if ModelFlags.translucent_twopass in combined_flags:
                LOGGER.warning(
                    'Both $mostlyopaque and $opaque set with models: {}',
                    {prop.model for prop in prop_pos}
                )
        elif ModelFlags.translucent_twopass in combined_flags:
            f.write('$mostlyopaque\n')

        if ModelFlags.no_forced_fade in combined_flags:
            f.write('$noforcedfade\n')

        if ModelFlags.ambient_boost in combined_flags:
            f.write('$ambientboost\n')

        if ModelFlags.do_not_cast_shadows in combined_flags:
            f.write('$donotcastshadows\n')

        for mat in sorted(cdmats):
            f.write('$cdmaterials "{}"\n'.format(mat))

        if coll_mesh.triangles:
            f.write(QC_COLL_TEMPLATE)
    LOGGER.debug('Wrote {}/model.qc', temp_folder)


async def build_reference(prop: PropPos, qc: QC, mdl: Model) -> Mesh:
    """Load and parse the reference SMD."""
    LOGGER.info('Parsing ref "{}#{}"', qc.ref_smd, prop.skin)
    async with LIM_PARSE:
        with open(qc.ref_smd, 'rb') as fb:
            mesh = await trio.to_thread.run_sync(Mesh.parse_smd, fb)

    if prop.skin != 0 and prop.skin < len(mdl.skins):
        # We need to rename the materials to match the skin.
        swap_skins = dict(zip(
            mdl.skins[0],
            mdl.skins[prop.skin]
        ))
        for tri in mesh.triangles:
            tri.mat = swap_skins.get(tri.mat, tri.mat)

    # For some reason all the SMDs are rotated badly, but only
    # if we append them.
    rot = Matrix.from_yaw(90)
    for tri in mesh.triangles:
        for vert in tri:
            vert.pos @= rot
            vert.norm @= rot
    return mesh


async def build_collision(qc: QC, prop: PropPos, ref_mesh: Mesh, needs_split: bool) -> List[Mesh]:
    """Get the correct collision mesh for this model."""
    if prop.solidity is CollType.NONE:  # Non-solid
        return []
    elif prop.solidity is CollType.VPHYS or prop.solidity is CollType.BSP:
        if qc.phy_smd is None:
            return []

        LOGGER.info('Parsing coll "{}"', qc.phy_smd)
        async with LIM_PARSE:
            with open(qc.phy_smd, 'rb') as fb:
                coll = await trio.to_thread.run_sync(Mesh.parse_smd, fb)

        rot = Matrix.from_yaw(90)
        for tri in coll.triangles:
            for vert in tri:
                vert.pos @= rot
                vert.norm @= rot

        if qc.is_concave and needs_split:
            return await trio.to_thread.run_sync(coll.split_collision)
        else:
            return [coll]

    # Else, it's one of the three bounding box types.
    # We don't really care about which.
    bbox_min, bbox_max = Vec.bbox(
        vert.pos
        for tri in
        ref_mesh.triangles
        for vert in tri
    )
    return [Mesh.build_bbox('static_prop', 'phy', bbox_min, bbox_max)]


def optimise_collision(
    tolerance: float,
    bone_link: List[Tuple[Bone, float]],
    coll_mesh: Mesh,
    groups: Dict[Mesh, float],
) -> None:
    """Attempt to merge together collision groups."""
    todo: Set[Mesh] = set(groups)
    # Pairs we know don't combine correctly.
    failures: Set[Tuple[Mesh, Mesh]] = set()
    zero_norm = Vec()
    while todo:
        mesh1 = todo.pop()
        for mesh2 in todo:
            if (mesh1, mesh2) in failures or (mesh2, mesh1) in failures:
                continue
            combined = Mesh(coll_mesh.bones, coll_mesh.animation, [
                Triangle(
                    'phys',
                    Vertex(v1, zero_norm, 0.0, 0.0, bone_link),
                    Vertex(v2, zero_norm, 0.0, 0.0, bone_link),
                    Vertex(v3, zero_norm, 0.0, 0.0, bone_link),
                )
                for v1, v2, v3 in quickhull(
                    vert.pos
                    for tri in itertools.chain(mesh1.triangles, mesh2.triangles)
                    for vert in tri
                )
            ])
            combined_vol = combined.compute_volume()
            diff = abs(groups[mesh1] + groups[mesh2] - combined_vol)
            LOGGER.debug('Volume diff: {}', diff)
            if diff < tolerance:
                todo.discard(mesh2)
                todo.add(combined)
                LOGGER.info('{} + {} -> {}', id(mesh1), id(mesh2), id(combined))
                groups[combined] = combined_vol
                break
            else:
                failures.add((mesh1, mesh2))
        else:
            # Failed against all, this is fully optimised.
            mesh1.smooth_normals()
            coll_mesh.triangles += mesh1.triangles


def load_qcs(qc_folder: Path) -> Iterator[Tuple[str, QC]]:
    """Parse through all the QC files to match to compiled models."""
    for dirpath, dirnames, filenames in os.walk(str(qc_folder)):
        qc_loc = Path(dirpath)
        for fname in filenames:
            if not fname.endswith('.qc'):
                continue
            qc_path = qc_loc / fname

            qc_result = parse_qc(qc_loc, qc_path)

            if qc_result is None:
                # It's a dynamic QC, we can't combine.
                continue

            (
                model_name, is_concave,
                ref_scale, ref_smd,
                phy_scale, phy_smd,
            ) = qc_result

            # We can't parse non-SMD files.
            if ref_smd.suffix.casefold() != '.smd':
                LOGGER.warning('Reference mesh not a SMD:\n{}', ref_smd)
                continue

            if phy_smd is not None and phy_smd.suffix.casefold() != '.smd':
                LOGGER.warning('Collision mesh not a SMD:\n{}', ref_smd)
                continue

            yield unify_mdl(model_name), QC(
                str(qc_path).replace('\\', '/'),
                str(ref_smd).replace('\\', '/'),
                str(phy_smd).replace('\\', '/') if phy_smd else None,
                ref_scale,
                phy_scale,
                is_concave,
            )


def parse_qc(qc_loc: Path, qc_path: Path) -> Optional[Tuple[
    str, bool,
    float, Path,
    float, Optional[Path],
]]:
    """Parse a single QC file."""
    model_name = ref_smd = phy_smd = None
    scale_factor = ref_scale = phy_scale = 1.0
    is_concave = False

    with open(str(qc_path)) as f:
        tok = Tokenizer(
            f, qc_path,
            allow_escapes=False,
            allow_star_comments=True,
        )
        for token_type, token_value in tok:
            if token_type is Token.STRING:
                token_value = token_value.casefold()
                if token_value == '$scale':
                    scale_factor = float(tok.expect(Token.STRING))
                elif token_value == '$modelname':
                    model_name = tok.expect(Token.STRING)
                elif token_value in ('$bodygroup', '$body', '$model'):
                    tok.expect(Token.STRING)  # group name.
                    body_type, body_value = tok()
                    if body_type is Token.STRING:
                        # $body name "file.smd"
                        if ref_smd:
                            # Multiple bodygroups, can't deal with that.
                            LOGGER.debug(
                                'QC "{}" has multiple bodygroups: {}, {}',
                                qc_path, ref_smd, (qc_loc /  body_value),
                            )
                            return None
                        else:
                            ref_smd = qc_loc / body_value
                            ref_scale = scale_factor
                        continue
                    elif body_type is Token.NEWLINE:
                        tok.expect(Token.BRACE_OPEN)
                    elif body_type is not Token.BRACE_OPEN:
                        raise tok.error(body_type)

                    for body_type, body_value in tok:
                        if body_type is Token.BRACE_CLOSE:
                            break
                        elif body_type is Token.STRING:
                            if body_value.casefold() == "studio":
                                if ref_smd:
                                    LOGGER.debug(
                                        'QC "{}" has multiple bodygroups: {}, {}',
                                        qc_path, ref_smd, tok.peek(),
                                    )
                                    return None
                                else:
                                    ref_smd = qc_loc / tok.expect(Token.STRING)
                                    ref_scale = scale_factor
                        elif body_type is not Token.NEWLINE:
                            raise tok.error(body_type)

                elif token_value in ('$collisionmodel', '$collisionjoints'):
                    phy_smd = qc_loc / tok.expect(Token.STRING)
                    phy_scale = scale_factor
                    next_typ, next_val = next(tok.skipping_newlines())
                    if next_typ is Token.BRACE_OPEN:
                        for body_value in tok.block(token_value, consume_brace=False):
                            if body_value.casefold() == '$concave':
                                is_concave = True
                    else:
                        tok.push_back(next_typ, next_val)

                # We can't support this.
                elif token_value in (
                    '$ikchain',
                    '$weightlist',
                    '$poseparameter',
                    '$proceduralbones',
                    '$jigglebone',
                    # Allow LOD models, propcombine is better than that.
                    # '$lod',
                ):
                    LOGGER.debug('QC "{}": Option {} is not supported', qc_path, token_value)
                    return None
            elif token_type is Token.BRACE_OPEN:
                # Skip other "compound" sections we don't care about.
                depth = 1
                for body_type, body_value in tok:
                    if body_type is Token.BRACE_CLOSE:
                        depth -= 1
                        if not depth:
                            break
                    elif body_type is Token.BRACE_OPEN:
                        depth += 1
                else:
                    raise tok.error("EOF reached without closing brace (})!")

    if model_name is None or ref_smd is None:
        # Malformed...
        LOGGER.warning('Cannot parse "{}"... ({}, {})', qc_path, model_name, ref_smd)
        return None

    return (
        model_name, is_concave,
        ref_scale, ref_smd,
        phy_scale, phy_smd,
    )


async def decompile_model(
    fsys: FileSystemChain,
    cache_loc: Path,
    crowbar: Path,
    filename: str,
    checksum: bytes,
) -> Optional[QC]:
    """Use Crowbar to decompile models directly for propcombining."""
    cache_folder = cache_loc / Path(filename).with_suffix('')
    info_path = cache_folder / 'info.kv'
    if cache_folder.exists():
        try:
            with info_path.open() as f:
                cache_kv = Keyvalues.parse(f).find_block('qc', or_blank=True)
            # Added later, remake if not present.
            if 'concave' not in cache_kv:
                raise FileNotFoundError
        except (FileNotFoundError, KeyValError):
            pass
        else:
            # Previous compilation.
            if checksum == bytes.fromhex(cache_kv['checksum', '']):
                ref_smd_name = cache_kv['ref', '']
                if not ref_smd_name:
                    return None
                phy_smd_name = cache_kv['phy', None]
                if phy_smd_name is not None:
                    phy_smd_name = str(cache_folder / phy_smd_name)
                return QC(
                    str(info_path),
                    str(cache_folder / ref_smd_name),
                    phy_smd_name,
                    cache_kv.float('ref_scale', 1.0),
                    cache_kv.float('phy_scale', 1.0),
                    cache_kv.bool('concave'),
                )
            # Otherwise, re-decompile.
    LOGGER.info('Decompiling {}...', filename)
    qc: Optional[QC] = None

    # Extract out the model to a temp dir.
    async with LIM_PROCESS:
        with TemporaryDirectory() as tempdir:
            stem = Path(filename).stem
            filename_no_ext = filename[:-4]
            for mdl_ext in MDL_EXTS:
                try:
                    file = fsys[filename_no_ext + mdl_ext]
                except FileNotFoundError:
                    pass
                else:
                    with file.open_bin() as src, Path(tempdir, stem + mdl_ext).open('wb') as dest:
                        shutil.copyfileobj(src, dest)
            LOGGER.debug('Extracted "{}" to "{}"', filename, tempdir)
            args = [
                str(crowbar), 'decompile',
                '-i', str(Path(tempdir, stem + '.mdl')),
                '-o', str(cache_folder),
            ]
            LOGGER.debug('Executing {}', ' '.join(args))
            result = await trio.run_process(args, capture_stdout=True, check=False)
            if result.returncode != 0:
                LOGGER.warning('Could not decompile "{}"!', filename)
                LOGGER.debug('{}', result.stdout.replace(b'\r\n', b'\n').decode('ascii', 'replace'))
                return None
    # There should now be a QC file here.
    for qc_path in cache_folder.glob('*.qc'):
        LOGGER.debug('Parse decompiled QC "{}"...', qc_path)
        qc_result = await trio.to_thread.run_sync(parse_qc, cache_folder, qc_path)
        break
    else:  # not found.
        LOGGER.warning('No QC outputted into {}', cache_folder)
        qc_result = None
        qc_path = Path()

    cache_kv = Keyvalues('qc', [])
    cache_kv['checksum'] = checksum.hex()

    if qc_result is not None:
        (
            model_name, is_concave,
            ref_scale, ref_smd,
            phy_scale, phy_smd,
        ) = qc_result
        qc = QC(
            str(qc_path).replace('\\', '/'),
            str(ref_smd).replace('\\', '/'),
            str(phy_smd).replace('\\', '/') if phy_smd else None,
            ref_scale,
            phy_scale,
            is_concave,
        )

        cache_kv['ref'] = Path(ref_smd).name
        cache_kv['ref_scale'] = format(ref_scale, '.6g')

        if phy_smd is not None:
            cache_kv['phy'] = Path(phy_smd).name
            cache_kv['phy_scale'] = format(phy_scale, '.6g')
        cache_kv['concave'] = bool_as_int(is_concave)
    else:
        cache_kv['ref'] = ''  # Mark as not present.

    with info_path.open('w') as f:
        for line in cache_kv.export():
            f.write(line)
    return qc


def group_props_ent(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    get_model: Callable[[str], Tuple[Optional[QC], Optional[Model]]],
    brush_models: MutableMapping[Entity, BModel],
    grouper_ents: List[Entity],
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, merge props according to the provided ents."""
    # Ents with group names. We have to split those by filter too.
    grouped_sets: Dict[Tuple[str, FrozenSet[str]], CombineVolume] = {}
    # Skinset filter -> volumes that match.
    sets_by_skin: Dict[FrozenSet[str], List[CombineVolume]] = defaultdict(list)

    empty_fs = frozenset('')

    for ent in grouper_ents:
        origin = Vec.from_str(ent['origin'])

        skinset = empty_fs
        mdl_name = ent['prop']
        if mdl_name:
            qc, mdl = get_model(mdl_name)
            if mdl is not None:
                skinset = frozenset({
                    tex.casefold().replace('\\', '/')
                    for tex in
                    mdl.iter_textures([conv_int(ent['skin'])])
                })

        angles = Angle.from_str(ent['angles'])

        # Group name
        group_name = ent['name']

        if group_name:
            try:
                combine_set = grouped_sets[group_name, skinset]
            except KeyError:
                combine_set = grouped_sets[group_name, skinset] = CombineVolume(group_name, skinset, origin)
                sets_by_skin[skinset].append(combine_set)
        else:
            combine_set = CombineVolume(group_name, skinset, origin)
            sets_by_skin[skinset].append(combine_set)

        if ent['classname'] == 'comp_propcombine_set':
            # Bounding box collision.
            mins, maxes = Vec.bbox(
                Vec.from_str(ent['mins']),
                Vec.from_str(ent['maxs']),
            )
            if combine_set.mins is None:
                combine_set.mins = origin + mins
            else:
                combine_set.mins.min(origin + mins)
            if combine_set.maxes is None:
                combine_set.maxes = origin + maxes
            else:
                combine_set.maxes.max(origin + maxes)

            size = maxes - mins
            # Enlarge slightly to ensure it never has a zero area.
            # This ensures items on the edge are included.
            mins -= 0.05
            maxes += 0.05
            combine_set.volume += size.x * size.y * size.z
            combine_set.collision.append(make_collision_bbox(origin, angles, mins, maxes))
        elif ent['classname'] == 'comp_propcombine_volume':
            # Brushwork collision. Pop from the dict, so the brush model is removed.
            try:
                brush = brush_models.pop(ent)
            except KeyError:
                raise ValueError(
                    f'No model for propcombine volume {repr(combine_set)} at '
                    f'{str(origin)}')
            # Use the bounding box as a volume approximation,
            # it's only needed for sorting the volumes.
            size = brush.maxes - brush.mins
            combine_set.volume += size.x * size.y * size.z
            combine_set.collision.append(make_collision_brush(origin, angles, brush))
            if combine_set.mins is None:
                combine_set.mins = brush.origin + brush.mins
            else:
                combine_set.mins.min(brush.origin + brush.mins)
            if combine_set.maxes is None:
                combine_set.maxes = brush.origin + brush.maxes
            else:
                combine_set.maxes.max(brush.origin + brush.maxes)
        else:
            raise AssertionError(ent['classname'])

    # We want to apply a ordering to groups, so smaller ones apply first, and
    # filtered ones override all others.
    for group_list in sets_by_skin.values():
        group_list.sort(key=operator.attrgetter('volume'))
    # Groups with no filter have no skins in the group.
    unfiltered_group = sets_by_skin.get(frozenset(), [])

    # Each of these groups cannot be merged with other ones.
    for group_key, group in prop_groups.items():
        if group_key is None:
            continue

        # No point merging single/empty groups.
        group_skinset = group_key[0]
        if len(group) < min_cluster:
            group.clear()
            continue

        for combine_set in itertools.chain(sets_by_skin.get(group_skinset, ()), unfiltered_group):
            found = []
            for prop in list(group):
                if combine_set.contains(prop.origin):
                    found.append(prop)
                    combine_set.used = True

            if not found:  # No point checking an empty list.
                continue

            actual: List[StaticProp] = []
            total_verts = 0
            for prop in found:
                qc, mdl = get_model(prop.model)
                assert mdl is not None
                total_verts += mdl.total_verts
                if total_verts > MAX_VERTS:
                    # Warn for groups since these were intentionally built.
                    LOGGER.warning(
                        'Hit vert limit in group {} with models {}', combine_set,
                        {prop.model for prop in found},
                    )
                    # Output this prop, then start a new group.
                    if len(actual) >= min_cluster:
                        yield list(actual)
                        for sub_prop in actual:
                            group.remove(sub_prop)
                    actual.clear()
                    total_verts = mdl.total_verts
                actual.append(prop)
            if len(actual) >= min_cluster:
                yield list(actual)
                for prop in actual:
                    group.remove(prop)

    # And log unused groups
    for combine_set_list in sets_by_skin.values():
        for combine_set in combine_set_list:
            if not combine_set.used:
                LOGGER.warning('Unused comp_propcombine_volume/_set {}', combine_set)


def group_props_auto(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    get_model: Callable[[str], Tuple[Optional[QC], Optional[Model]]],
    min_dist: float,
    max_dist: float,
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, automatically find close props to merge."""
    min_dist_sq = min_dist * min_dist
    max_dist_sq = max_dist * max_dist
    neighbours: Dict[StaticProp, Sequence[StaticProp]] = {}

    def find_neighbours(start: StaticProp) -> Sequence[StaticProp]:
        """Find props within dist from the specified one."""
        try:
            return neighbours[start]
        except KeyError:
            pass
        neigh = [
            prop for prop in group
            if (prop.origin - start.origin).mag_sq() <= min_dist_sq
        ]
        neighbours[start] = neigh
        return neigh

    UNSET: Literal['unset'] = 'unset'
    NOISE: Literal['noise'] = 'noise'

    # Each of these groups cannot be merged with other ones.
    for group in prop_groups.values():
        # No point merging single/empty groups.
        if len(group) < 2:
            continue

        # DBSCAN algorithm.
        labels: Dict[StaticProp, Union[int, Literal['noise', 'unset']]] = dict.fromkeys(group, UNSET)
        neighbours.clear()
        cluster_ind = 0

        LOGGER.debug('Grouping {} props', len(group))

        for prop in group:
            if labels[prop] is not UNSET:
                continue
            neigh = find_neighbours(prop)
            if len(neigh) < min_cluster:
                labels[prop] = NOISE
                continue
            cluster_ind += 1
            labels[prop] = cluster_ind
            todo = set(neigh)

            while todo:
                sub_prop = todo.pop()
                if labels[sub_prop] is NOISE:
                    labels[sub_prop] = cluster_ind
                elif labels[sub_prop] != UNSET:
                    continue  # Already handled.
                labels[sub_prop] = cluster_ind
                neigh = find_neighbours(sub_prop)
                if len(neigh) > min_cluster:
                    todo.update(neigh)

        neighbours.clear()  # Discard, no longer useful.

        clusters: Dict[int, List[StaticProp]] = defaultdict(list)
        for prop, key in labels.items():
            if type(key) is int:
                clusters[key].append(prop)

        # We now have many potential groups, which may be extremely large.
        # We want to split these up, so they don't extend too far.
        for cluster in clusters.values():
            warned: bool = False
            todo = set(cluster)
            while len(todo) > min_cluster:
                # First find the prop the furthest from the center-point.
                average_pos = sum((prop.origin for prop in todo), Vec()) / len(todo)
                central_prop = max(todo, key=lambda prop: (prop.origin - average_pos).mag_sq())

                total_verts = 0
                selected_props: List[StaticProp] = []
                found_matches = False
                for prop in list(todo):
                    # Exceeds the max radius?
                    if (prop.origin - central_prop.origin).mag_sq() > max_dist_sq:
                        continue
                    qc, mdl = get_model(prop.model)
                    assert mdl is not None
                    total_verts += mdl.total_verts
                    if total_verts > MAX_VERTS:
                        # Make this just info level, just might be props nearby.
                        if not warned:
                            bb_min, bb_max = Vec.bbox(prop.origin for prop in cluster)
                            LOGGER.info(
                                'Hit vert limit for auto group @ ({} - {}) with models {}' ,
                                bb_min, bb_max,
                                {prop.model for prop in cluster},
                            )
                            warned = True
                        # Split the group here, create a new prop.
                        if len(selected_props) >= min_cluster:
                            found_matches = True
                            todo.difference_update(selected_props)
                            yield selected_props
                        selected_props = []
                        total_verts = mdl.total_verts
                    selected_props.append(prop)

                if len(selected_props) >= min_cluster:
                    yield selected_props
                    todo.difference_update(selected_props)
                    found_matches = True
                if not found_matches:
                    # The selected prop was too far away to cluster. Discard it, so we pick a
                    # different one. It should be added by itself, it's on its own mostly.
                    todo.discard(central_prop)
            # Once the while loop terminates, our group is too small to actually cluster any more.
            # The main combine() function will re-add them to the map automatically.


async def combine(
    bsp: BSP,
    bsp_ents: VMF,
    pack: PackList,
    game: Game,
    studiomdl_loc: Path,
    *,
    qc_folders: Optional[List[Path]]=None,
    crowbar_loc: Optional[Path]=None,
    decomp_cache_loc: Optional[Path]=None,
    compile_dump: Optional[Path]=None,
    blacklist: Iterable[str]=(),
    min_auto_range: float=0.0,
    max_auto_range: float=math.inf,
    min_cluster: int=2,
    min_cluster_auto: int=0,
    volume_tolerance: float=1.0,
    debug_dump: bool=False,
    pack_models: bool=True,
) -> None:
    """Combine props in this map."""
    LOGGER.debug(
        'Propcombine: decomp cache={}, crowbar={}, studiomdl={}',
        decomp_cache_loc, crowbar_loc, studiomdl_loc,
    )

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot propcombine!')
        return

    # Convert the blacklist into a regex, for fast comparison.
    blacklist_re = re.compile('|'.join(sorted([
        '(?s:%)'.replace(
            '%', fnmatch.translate(mdl_name.casefold().replace("\\", "/")),
        )
        for mdl_name in blacklist
    ], key=len)))

    if not qc_folders and decomp_cache_loc is None:
        # If gameinfo is blah/game/hl2/gameinfo.txt,
        # QCs should be in blah/content/ according to Valve's scheme.
        # But allow users to override this.
        # If Crowbar's path is provided, that means they may want to just supply nothing.
        qc_folders = [game.path.parent.parent / 'content']

    # Parse through all the QC files.
    qc_map: Dict[str, Union[QC, None]] = {}
    if qc_folders:
        LOGGER.info('Parsing QC files. Paths: \n{}', '\n'.join(map(str, qc_folders)))
        for qc_folder in qc_folders:
            for mdl_name, loaded_qc in load_qcs(qc_folder):
                qc_map[mdl_name] = loaded_qc
        LOGGER.info('Done! {} prop QCs found.', len(qc_map))

    map_name = Path(bsp.filename).stem

    # Holds the QC and mdl for a prop, if available.
    mdl_map: Dict[str, Tuple[QC, Model]] = {}
    # Wipe these, if they're being used again.
    _mesh_cache.clear()
    _coll_cache.clear()
    missing_qcs: Set[str] = set()

    async def load_model(key: str, filename: str) -> None:
        """Given a filename, load/parse the QC and MDL data."""
        if blacklist_re.fullmatch(key) is not None:
            LOGGER.debug('Model {} was blacklisted.', filename)
            return
        if filename.endswith(('.glb', '.gltf')):
            # Can't support these yet.
            return
        try:
            mdl_file = pack.fsys[filename]
        except FileNotFoundError:
            # We don't have this model, we can't combine...
            LOGGER.debug('Model {} was not found in the filesystem.', filename)
            return

        model = await trio.to_thread.run_sync(Model, pack.fsys, mdl_file)
        if 'no_propcombine' in model.keyvalues.casefold():
            LOGGER.debug('Model {} is blacklisted in the QC.', filename)
            return

        if model is None:
            return

        try:
            qc = qc_map[key]
        except KeyError:
            if crowbar_loc is None or decomp_cache_loc is None:
                LOGGER.debug('Model {} has no QC!', filename)
                missing_qcs.add(filename)
                return
            qc = await decompile_model(pack.fsys, decomp_cache_loc, crowbar_loc, filename, model.checksum)

        if qc is not None:
            mdl_map[key] = qc, model

    async with trio.open_nursery() as nursery:
        # Dict to deduplicate.
        for key_, filename_ in {
            # Hammer allows filenames like "/blah/", we don't want it to be looking at the root.
            unify_mdl(prop.model): prop.model.lstrip('/\\')
            for prop in bsp.props
        }.items():
            nursery.start_soon(load_model, key_, filename_)

    def get_model(filename: str) -> Union[Tuple[QC, Model], Tuple[None, None]]:
        """Fetch the parsed model and QC, or None if not possible."""
        try:
            return mdl_map[unify_mdl(filename)]
        except KeyError:
            return None, None

    # Ignore this, we handle lighting origin ourselves.
    relevant_flags = ~StaticPropFlags.HAS_LIGHTING_ORIGIN

    def get_grouping_key(prop: StaticProp) -> Optional[tuple]:
        """Compute a grouping key for this prop.

        Only props with matching key can be possibly combined.
        If None it cannot be combined.
        """
        try:
            qc, model = mdl_map[unify_mdl(prop.model)]
        except KeyError:
            return None

        return (
            # Must be first, we pull this out later.
            frozenset({
                tex.casefold().replace('\\', '/')
                for tex in
                model.iter_textures([prop.skin])
            }),
            (prop.flags & relevant_flags).value,
            # Do not allow combining across an areaportal boundary.
            frozenset({leaf.area for leaf in prop.visleafs}),
            model.contents,
            model.surfaceprop,
            prop.renderfx,
            *prop.tint,
        )

    prop_count = 0

    # First, construct groups of props that can possibly be combined.
    prop_groups: Dict[Optional[tuple], List[StaticProp]] = defaultdict(list)
    for prop in bsp.props:
        prop_groups[get_grouping_key(prop)].append(prop)
        prop_count += 1

    # This holds the list of all props we want in the map at the end.
    final_props: List[StaticProp] = []
    grouper: Iterator[List[StaticProp]]
    grouper_ents = list(bsp_ents.by_class['comp_propcombine_set'] | bsp_ents.by_class['comp_propcombine_volume'])
    if min_cluster_auto <= 2:
        min_cluster_auto = min_cluster
    if grouper_ents and min_auto_range > 0:
        LOGGER.info('{} propcombine sets present and auto-grouping enabled, combining...', len(grouper_ents))
        # Do ents first, that removes values from the lists in prop_groups,
        # then the auto grouper handles that.
        grouper = itertools.chain(
            group_props_ent(
                prop_groups,
                get_model,
                bsp.bmodels, grouper_ents,
                min_cluster,
            ),
            group_props_auto(
                prop_groups,
                get_model,
                min_auto_range, max_auto_range,
                min_cluster_auto or min_cluster,
            )
        )
    elif grouper_ents:
        LOGGER.info('Propcombine sets present ({}), combining...', len(grouper_ents))
        grouper = group_props_ent(
            prop_groups,
            get_model,
            bsp.bmodels, grouper_ents,
            min_cluster,
        )
    elif min_auto_range > 0:
        LOGGER.info('Automatically finding propcombine sets...')
        grouper = group_props_auto(
            prop_groups,
            get_model,
            min_auto_range, max_auto_range,
            min_cluster_auto or min_cluster,
        )
    else:
        # No way provided to choose props.
        LOGGER.info('No propcombine groups or range provided.')
        return

    # These are models we cannot merge no matter what -
    # no source files etc.
    cannot_merge = prop_groups.pop(None, [])
    final_props.extend(cannot_merge)

    LOGGER.debug('Prop groups: \n{}', '\n'.join([
        f'{group}: {len(props)}'
        for group, props in
        sorted(prop_groups.items(), key=operator.itemgetter(0))
    ]))

    # Create a set of every prop, then remove the ones we group.
    # We'll then be left with props we didn't group and so should persist.
    rejected = set(itertools.chain.from_iterable(prop_groups.values()))
    group_count = 0
    compiler: PropCombiner
    with PropCombiner(
        game,
        studiomdl_loc,
        pack,
        map_name,
        folder_name='propcombine',
        version={
            'ver': 3,  # This is bumped if all models need to be recompiled.
            'vol_tolerance': volume_tolerance,
        },
        compile_dir=compile_dump,
        pack_models=pack_models,
    ) as compiler:
        async def do_combine(group: List[StaticProp]) -> None:
            """Task run to combine one prop."""
            grouped_prop = await combine_group(compiler, group, get_model, volume_tolerance)
            rejected.difference_update(group)
            final_props.append(grouped_prop)

        async with trio.open_nursery() as nursery:
            for group_ in grouper:
                nursery.start_soon(do_combine, group_)
                group_count += 1

    final_props.extend(rejected)

    if debug_dump:
        dump_vmf = VMF()
        for prop in rejected:
            dump_vmf.create_ent(
                'prop_static',
                skin=prop.skin,
                model=prop.model,
                origin=prop.origin,
                angles=prop.angles,
                solid=prop.solidity,
                rendercolor=prop.tint,
            )
        dump_fname = Path(bsp.filename).with_name(map_name + '_propcombine_reject.vmf')
        LOGGER.info('Dumping uncombined props to {}...', dump_fname)
        with dump_fname.open('w') as f:
            dump_vmf.export(f)

    LOGGER.info(
        'Combined {} props into {} groups ({} this compile):\n'
        ' - {} grouped models\n'
        ' - {} ineligible\n'
        ' - {} failed to combine',
        prop_count,
        len(final_props),
        compiler.built_count,
        group_count,
        len(cannot_merge),
        len(rejected),
    )
    LOGGER.debug('Models with unknown QCs: \n{}', '\n'.join(sorted(missing_qcs)))
    # If present, delete old cache file. We'll have cleaned up the models.
    try:
        os.remove(compiler.model_folder_abs / 'cache.vdf')
    except FileNotFoundError:
        pass

    bsp.props = final_props
