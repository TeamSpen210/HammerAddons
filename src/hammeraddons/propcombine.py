"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import fnmatch
import operator
import os
import random
import colorsys
import re
import shutil
import subprocess
import itertools
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Optional, Tuple, Callable, NamedTuple,
    FrozenSet, Dict, List, Set,
    Iterator, Union, MutableMapping, Iterable,
)
from srctools._math import quickhull

from srctools import (
    Vec, VMF, Entity, conv_int, Angle, Matrix, FileSystemChain,
    Property, KeyValError, bool_as_int,
)
from srctools.tokenizer import Tokenizer, Token
from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp, StaticPropFlags, BModel, VisLeaf
from srctools.mdl import Model, MDL_EXTS
from srctools.smd import Mesh, Triangle, Vertex
from srctools.compiler.mdl_compiler import ModelCompiler


LOGGER = get_logger(__name__)


class QC(NamedTuple):
    path: str  # QC path.
    ref_smd: str  # Location of main visible geometry.
    phy_smd: Optional[str]  # Relative location of collision model, or None
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

MAX_GROUP = 24  # Studiomdl does't allow more than this...

# Cache of the SMD models we have already parsed, so we don't need
# to parse them again. For the collision model, we store them pre-split.
_mesh_cache = {}  # type: Dict[Tuple[QC, int], Mesh]
_coll_cache = {}  # type: Dict[str, List[Mesh]]


def unify_mdl(path: str):
    """Compute a 'canonical' path for a given model."""
    path = path.casefold().replace('\\', '/')
    if not path.startswith('models/'):
        path = 'models/' + path
    if not path.endswith('.mdl'):
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
        self.mins = Vec()
        self.maxes = Vec()
        # Each volume in the group, specifying its collision behaviour.
        self.collision: List[Callable[[Vec], bool]] = []

        if group_name:
            self.desc = f'group "{group_name}"'
        else:
            self.desc = f'at {origin}'

    def contains(self, point: Vec) -> bool:
        """Check if the volume contains this point."""
        return any(coll(point) for coll in self.collision)


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
    # brushes = {
    #     br for leaf in brush.node.iter_leafs()
    #     for br in leaf.brushes
    # }

    def check(point: Vec) -> bool:
        """Check if the given position is inside the volume."""
        local_point = (point - origin) @ inv_angles
        leaf = brush.node.test_point(local_point)
        return leaf is not None and len(leaf.brushes) > 0
    return check


class CollType(Enum):
    """Collision types that static props can have."""
    NONE = 0  # No collision
    BSP = 1  # Treat the same as MODEL.
    BBOX = 2
    OBB = 3
    OBB_YAW = 4
    VPHYS = 6  # Collision model


class PropPos(NamedTuple):
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
    scale: float
    solidity: CollType


def combine_group(
    compiler: ModelCompiler,
    props: List[StaticProp],
    lookup_model: Callable[[str], Tuple[QC, Model]],
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
    avg_yaw = 0.0

    visleafs: Set[VisLeaf] = set()

    for prop in props:
        avg_pos += prop.origin
        avg_yaw += prop.angles.yaw
        visleafs.update(prop.visleafs)

    # Snap to nearest 15 degrees to keep the models themselves not
    # strangely rotated.
    avg_yaw = round(avg_yaw / (15 * len(props))) * 15.0
    avg_pos /= len(props)
    yaw_rot = Matrix.from_yaw(-avg_yaw)

    prop_pos = set()
    for prop in props:
        origin = round((prop.origin - avg_pos) @ yaw_rot, 7)
        angles = round(Vec(prop.angles), 7)
        angles.y -= avg_yaw
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
        prop_pos.add(PropPos(
            origin.x, origin.y, origin.z,
            angles.x, angles.y, angles.z,
            prop.model,
            mdl.checksum,
            prop.skin,
            prop.scaling,
            coll,
        ))
    # We don't want to build collisions if it's not used.
    has_coll = any(pos.solidity is not CollType.NONE for pos in prop_pos)
    mdl_name, result = compiler.get_model(
        (frozenset(prop_pos), has_coll),
        compile_func, (lookup_model, volume_tolerance),
    )

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model=mdl_name,
        origin=avg_pos,
        angles=Angle(0, avg_yaw - 90, 0),
        scaling=1.0,
        visleafs=visleafs,
        solidity=(CollType.VPHYS if has_coll else CollType.NONE).value,
        flags=props[0].flags,
        lighting=avg_pos,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
    )


def compile_func(
    mdl_key: Tuple[FrozenSet[PropPos], bool],
    temp_folder: Path,
    mdl_name: str,
    args: Tuple[Callable[[str], Tuple[QC, Model]], float],
) -> None:
    """Build this merged model."""
    LOGGER.info('Compiling {}...', mdl_name)
    prop_pos, has_coll = mdl_key
    lookup_model, volume_tolerance = args

    # Unify these properties.
    surfprops = set()  # type: Set[str]
    cdmats = set()  # type: Set[str]
    contents = set()  # type: Set[int]

    for prop in prop_pos:
        qc, mdl = lookup_model(prop.model)
        assert mdl is not None, prop.model
        surfprops.add(mdl.surfaceprop.casefold())
        cdmats.update(mdl.cdmaterials)
        contents.add(mdl.contents)

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
    coll_groups: dict[Mesh, float] = {}

    for prop in prop_pos:
        qc, mdl = lookup_model(prop.model)
        try:
            child_ref = _mesh_cache[qc, prop.skin]
        except KeyError:
            LOGGER.info('Parsing ref "{}"', qc.ref_smd)
            with open(qc.ref_smd, 'rb') as fb:
                child_ref = Mesh.parse_smd(fb)

            if prop.skin != 0 and prop.skin < len(mdl.skins):
                # We need to rename the materials to match the skin.
                swap_skins = dict(zip(
                    mdl.skins[0],
                    mdl.skins[prop.skin]
                ))
                for tri in child_ref.triangles:
                    tri.mat = swap_skins.get(tri.mat, tri.mat)

            # For some reason all the SMDs are rotated badly, but only
            # if we append them.
            rot = Matrix.from_yaw(90)
            for tri in child_ref.triangles:
                for vert in tri:
                    vert.pos @= rot
                    vert.norm @= rot

            _mesh_cache[qc, prop.skin] = child_ref

        child_coll = build_collision(qc, prop, child_ref)

        offset = Vec(prop.x, prop.y, prop.z)
        matrix = Matrix.from_angle(Angle(prop.pit, prop.yaw, prop.rol))

        ref_mesh.append_model(child_ref, matrix, offset, prop.scale * qc.ref_scale)

        if has_coll and child_coll is not None:
            scale = prop.scale * qc.phy_scale
            group = Mesh(coll_mesh.bones, coll_mesh.animation, [])
            for part in child_coll:
                for orig_tri in part.triangles:
                    new_tri = orig_tri.copy()
                    for vert in new_tri:
                        vert.links[:] = bone_link
                        vert.norm @= matrix
                        vert.pos *= scale
                        vert.pos.localise(offset, matrix)
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
            LOGGER.info('Optimising collisions:')
            # Attempt to merge together collision groups.
            todo: set[Mesh] = set(coll_groups)
            # Pairs we know don't combine correctly.
            failures: set[tuple[Mesh, Mesh]] = set()
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
                    diff = abs(coll_groups[mesh1] + coll_groups[mesh2] - combined_vol)
                    LOGGER.debug('Volume diff: {}', diff)
                    if diff < volume_tolerance:
                        todo.discard(mesh2)
                        todo.add(combined)
                        LOGGER.info('{} + {} -> {}', id(mesh1), id(mesh2), id(combined))
                        coll_groups[combined] = combined_vol
                        break
                    else:
                        failures.add((mesh1, mesh2))
                else:
                    # Failed against all, this is fully optimised.
                    mesh1.smooth_normals()
                    coll_mesh.triangles += mesh1.triangles
            LOGGER.info('Done.')
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

        for mat in sorted(cdmats):
            f.write('$cdmaterials "{}"\n'.format(mat))

        if coll_mesh.triangles:
            f.write(QC_COLL_TEMPLATE)


def build_collision(qc: QC, prop: PropPos, ref_mesh: Mesh) -> List[Mesh]:
    """Get the correct collision mesh for this model."""
    if prop.solidity is CollType.NONE:  # Non-solid
        return []
    elif prop.solidity is CollType.VPHYS or prop.solidity is CollType.BSP:
        if qc.phy_smd is None:
            return []
        try:
            return _coll_cache[qc.phy_smd]
        except KeyError:
            LOGGER.info('Parsing coll "{}"', qc.phy_smd)
            with open(qc.phy_smd, 'rb') as fb:
                coll = Mesh.parse_smd(fb)

            rot = Matrix.from_yaw(90)
            for tri in coll.triangles:
                for vert in tri:
                    vert.pos @= rot
                    vert.norm @= rot

            if qc.is_concave:
                coll_group = coll.split_collision()
            else:
                coll_group = [coll]
            _coll_cache[qc.phy_smd] = coll_group
            return coll_group
    # Else, it's one of the three bounding box types.
    # We don't really care about which.
    bbox_min, bbox_max = Vec.bbox(
        vert.pos
        for tri in
        ref_mesh.triangles
        for vert in tri
    )
    return [Mesh.build_bbox('static_prop', 'phy', bbox_min, bbox_max)]


def load_qcs(qc_map: Dict[str, QC], qc_folder: Path) -> None:
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

            # We can't parse FBX files right now.
            if ref_smd.suffix.casefold() not in ('.smd', '.dmx_DISABLE'):
                LOGGER.warning('Reference mesh not a SMD/DMX:\n{}', ref_smd)
                continue

            if phy_smd is not None and phy_smd.suffix.casefold() not in ('.smd', '.dmx_DISABLE'):
                LOGGER.warning('Collision mesh not a SMD/DMX:\n{}', ref_smd)
                continue

            qc_map[unify_mdl(model_name)] = QC(
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
                                    return None
                                else:
                                    ref_smd = qc_loc / tok.expect(Token.STRING)
                                    ref_scale = scale_factor
                        elif body_type is not Token.NEWLINE:
                            raise tok.error(body_type)

                elif token_value == '$collisionmodel':
                    phy_smd = qc_loc / tok.expect(Token.STRING)
                    phy_scale = scale_factor
                    next_typ, next_val = next(tok.skipping_newlines())
                    if next_typ is Token.BRACE_OPEN:
                        for body_value in tok.block('$collisionmodel', consume_brace=False):
                            if body_value.casefold() == '$concave':
                                is_concave = True
                    else:
                        tok.push_back(next_typ, next_val)

                # We can't support this.
                elif token_value in (
                    '$collisionjoints',
                    '$ikchain',
                    '$weightlist',
                    '$poseparameter',
                    '$proceduralbones',
                    '$jigglebone',
                    # Allow LOD models, propcombine is better than that.
                    # '$lod',
                ):
                    return None
            elif token_type is Token.BRACE_OPEN:
                # Skip other "compound" sections we don't care about.
                for body_type, body_value in tok:
                    if body_type is Token.BRACE_CLOSE:
                        break
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


def decompile_model(
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
                cache_props = Property.parse(f).find_block('qc', or_blank=True)
            # Added later, remake if not present.
            if 'concave' not in cache_props:
                raise FileNotFoundError
        except (FileNotFoundError, KeyValError):
            pass
        else:
            # Previous compilation.
            if checksum == bytes.fromhex(cache_props['checksum', '']):
                ref_smd = cache_props['ref', '']
                if not ref_smd:
                    return None
                phy_smd = cache_props['phy', None]
                if phy_smd is not None:
                    phy_smd = str(cache_folder / phy_smd)
                return QC(
                    str(info_path),
                    str(cache_folder / ref_smd),
                    phy_smd,
                    cache_props.float('ref_scale', 1.0),
                    cache_props.float('phy_scale', 1.0),
                    cache_props.bool('concave'),
                )
            # Otherwise, re-decompile.
    LOGGER.info('Decompiling {}...', filename)
    qc: Optional[QC] = None

    # Extract out the model to a temp dir.
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
        result = subprocess.run(args)
        if result.returncode != 0:
            LOGGER.warning('Could not decompile "{}"!', filename)
            return None
    # There should now be a QC file here.
    for qc_path in cache_folder.glob('*.qc'):
        qc_result = parse_qc(cache_folder, qc_path)
        break
    else:  # not found.
        LOGGER.warning('No QC outputted into {}', cache_folder)
        qc_result = None
        qc_path = Path()

    cache_props = Property('qc', [])
    cache_props['checksum'] = checksum.hex()

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

        cache_props['ref'] = Path(ref_smd).name
        cache_props['ref_scale'] = format(ref_scale, '.6g')

        if phy_smd is not None:
            cache_props['phy'] = Path(phy_smd).name
            cache_props['phy_scale'] = format(phy_scale, '.6g')
        cache_props['concave'] = bool_as_int(is_concave)
    else:
        cache_props['ref'] = ''  # Mark as not present.

    with info_path.open('w') as f:
        for line in cache_props.export():
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

            actual = set(found).intersection(group)
            if len(actual) >= min_cluster:
                yield list(actual)
                for prop in actual:
                    group.remove(prop)

    # And log unused groups
    for combine_set_list in sets_by_skin.values():
        for combine_set in combine_set_list:
            if not combine_set.used:
                LOGGER.warning('Unused comp_propcombine_set {}', combine_set.desc)


def group_props_auto(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    dist: float,
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, automatically find close props to merge."""
    # Each of these groups cannot be merged with other ones.

    dist_sq = dist * dist
    large_dist_sq = 4 * dist_sq

    for group in prop_groups.values():
        # No point merging single/empty groups.
        if len(group) < 2:
            continue

        todo = set(group)
        while todo:
            center = todo.pop()
            cluster = {center}

            for prop in todo:
                if (center.origin - prop.origin).mag_sq() <= large_dist_sq:
                    cluster.add(prop)
                    if len(cluster) > MAX_GROUP:
                        # Limit the number of maximum props that can be used.
                        break

            if len(cluster) < min_cluster:
                continue

            bbox_min, bbox_max = Vec.bbox(prop.origin for prop in cluster)
            center_pos = (bbox_min + bbox_max) / 2

            cluster_list: list[tuple[StaticProp, float]] = []

            for prop in cluster:
                prop_off = (center_pos - prop.origin).mag_sq()
                if prop_off <= dist_sq:
                    cluster_list.append((prop, prop_off))

            cluster_list.sort(key=operator.itemgetter(1))
            selected_props = [
                prop for prop, off in
                cluster_list[:MAX_GROUP]
            ]
            todo.difference_update(selected_props)

            if len(selected_props) >= min_cluster:
                yield selected_props


def combine(
    bsp: BSP,
    bsp_ents: VMF,
    pack: PackList,
    game: Game,
    studiomdl_loc: Path,
    *,
    qc_folders: List[Path]=None,
    crowbar_loc: Optional[Path]=None,
    decomp_cache_loc: Path=None,
    blacklist: Iterable[str]=(),
    auto_range: float=0,
    min_cluster: int=2,
    volume_tolerance: float=1.0,
    debug_tint: bool=False,
    debug_dump: bool=False,
) -> None:
    """Combine props in this map."""
    # First parse out the bbox and volume ents, so they are always removed.
    grouper_ents = list(bsp_ents.by_class['comp_propcombine_set'] | bsp_ents.by_class['comp_propcombine_volume'])

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot propcombine!')
        for ent in grouper_ents:
            ent.remove()
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
    qc_map: Dict[str, Optional[QC]] = {}
    if qc_folders:
        LOGGER.info('Parsing QC files. Paths: \n{}', '\n'.join(map(str, qc_folders)))
        for qc_folder in qc_folders:
            load_qcs(qc_map, qc_folder)
        LOGGER.info('Done! {} prop QCs found.', len(qc_map))

    map_name = Path(bsp.filename).stem

    # Don't re-parse models continually.
    mdl_map: Dict[str, Optional[Model]] = {}
    # Wipe these, if they're being used again.
    _mesh_cache.clear()
    _coll_cache.clear()
    missing_qcs: Set[str] = set()

    def get_model(filename: str) -> Union[Tuple[QC, Model], Tuple[None, None]]:
        """Given a filename, load/parse the QC and MDL data.

        Either both are returned, or neither are.
        """
        key = unify_mdl(filename)
        try:
            model = mdl_map[key]
        except KeyError:
            if blacklist_re.fullmatch(key) is not None:
                mdl_map[key] = qc_map[key] = None
                return None, None
            try:
                mdl_file = pack.fsys[filename]
            except FileNotFoundError:
                # We don't have this model, we can't combine...
                return None, None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)
            if 'no_propcombine' in model.keyvalues.casefold():
                mdl_map[key] = qc_map[key] = None
                return None, None

        if model is None or key in missing_qcs:
            return None, None

        try:
            qc = qc_map[key]
        except KeyError:
            if crowbar_loc is None:
                missing_qcs.add(key)
                return None, None
            qc = decompile_model(pack.fsys, decomp_cache_loc, crowbar_loc, filename, model.checksum)
            qc_map[key] = qc

        if qc is None:
            return None, None
        else:
            return qc, model

    # Ignore these two, they don't affect our new prop.
    relevant_flags = ~(StaticPropFlags.HAS_LIGHTING_ORIGIN | StaticPropFlags.DOES_FADE)

    def get_grouping_key(prop: StaticProp) -> Optional[tuple]:
        """Compute a grouping key for this prop.

        Only props with matching key can be possibly combined.
        If None it cannot be combined.
        """
        qc, model = get_model(prop.model)

        if model is None or qc is None:
            return None

        return (
            # Must be first, we pull this out later.
            frozenset({
                tex.casefold().replace('\\', '/')
                for tex in
                model.iter_textures([prop.skin])
            }),
            model.flags.value,
            (prop.flags & relevant_flags).value,
            model.contents,
            model.surfaceprop,
            prop.renderfx,
            *prop.tint,
        )

    prop_count = 0

    # First, construct groups of props that can possibly be combined.
    prop_groups: dict[Optional[tuple], list[StaticProp]] = defaultdict(list)

    # This holds the list of all props we want in the map at the end.
    final_props: list[StaticProp] = []

    if grouper_ents and auto_range > 0:
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
                auto_range,
                min_cluster,
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
    elif auto_range > 0:
        LOGGER.info('Automatically finding propcombine sets...')
        grouper = group_props_auto(
            prop_groups,
            auto_range,
            min_cluster,
        )
    else:
        # No way provided to choose props.
        LOGGER.info('No propcombine groups provided.')
        return

    for prop in bsp.props:
        prop_groups[get_grouping_key(prop)].append(prop)
        prop_count += 1

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
    with ModelCompiler(
        game,
        studiomdl_loc,
        pack,
        map_name,
        'propcombine',
        version={
            'ver': 1,
            'vol_tolerance': volume_tolerance,
        },
    ) as compiler:
        for group in grouper:
            grouped_prop = combine_group(compiler, group, get_model, volume_tolerance)
            rejected.difference_update(group)
            if debug_tint:
                # Compute a random hue, and convert back to RGB 0-255.
                r, g, b = colorsys.hsv_to_rgb(random.random(), 1, 1)
                grouped_prop.tint = Vec(round(r*255), round(g*255), round(b*255))
            final_props.append(grouped_prop)
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
        'Combined {} props into {}:\n - {} grouped models\n - {} ineligable\n - {} failed to combine',
        prop_count,
        len(final_props),
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
