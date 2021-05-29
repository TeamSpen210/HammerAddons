"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import operator
import os
import random
import colorsys
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
    Iterator, Union,
)

from srctools import (
    Vec, VMF, Entity, conv_int, Angle, Matrix, FileSystemChain,
    Property, KeyValError,
)
from srctools.tokenizer import Tokenizer, Token
from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp, StaticPropFlags
from srctools.mdl import Model, MDL_EXTS
from srctools.smd import Mesh
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
# to parse them again. The second is the collision model.
_mesh_cache = {}  # type: Dict[Tuple[QC, int], Mesh]
_coll_cache = {}  # type: Dict[str, Mesh]


def unify_mdl(path: str):
    """Compute a 'canonical' path for a given model."""
    path = path.casefold().replace('\\', '/')
    if not path.startswith('models/'):
        path = 'models/' + path
    if not path.endswith('.mdl'):
        path = path + '.mdl'
    return path


class CombineVolume:
    """Parsed comp_propcombine_sets."""
    def __init__(self, group_name: str, skinset: FrozenSet, origin: Vec) -> None:
        self.group = group_name
        self.skinset = skinset
        self.volume = 0.0  # For sorting
        self.used = False
        # To do collision checks, for each volume construct a list of planes.
        # Then we can check if a prop is inside any of those.
        self.collision: List[List[Tuple[Vec, Vec]]] = []
        if group_name:
            self.desc = f'group "{group_name}"'
        else:
            self.desc = f'at {origin}'

    def contains(self, point: Vec) -> bool:
        """Check if the given position is inside the volume."""
        for convex in self.collision:
            for pos, norm in convex:
                off = pos - point
                # This is the actual distance, so we'll use a rather large
                # "epsilon" to catch objects close to the edges.
                if Vec.dot(off, norm) < -0.1:
                    break  # Outside a plane, it doesn't match this convex.
            else:  # Inside all these planes, it's inside.
                return True
        return False  # All failed, not present.


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
    skin: int
    scale: float
    solidity: CollType


def combine_group(
    compiler: ModelCompiler,
    props: List[StaticProp],
    lookup_model: Callable[[str], Tuple[QC, Model]],
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

    visleafs = set()  # type: Set[int]

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
        prop_pos.add(PropPos(
            origin.x, origin.y, origin.z,
            angles.x, angles.y, angles.z,
            prop.model,
            prop.skin,
            prop.scaling,
            coll,
        ))
    # We don't want to build collisions if it's not used.
    has_coll = any(pos.solidity is not CollType.NONE for pos in prop_pos)
    mdl_name, result = compiler.get_model(
        (frozenset(prop_pos), has_coll),
        compile_func, lookup_model,
    )

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model=mdl_name,
        origin=avg_pos,
        angles=Angle(0, avg_yaw - 90, 0),
        scaling=1.0,
        visleafs=sorted(visleafs),
        solidity=(CollType.VPHYS if has_coll else CollType.NONE).value,
        flags=props[0].flags,
        lighting_origin=avg_pos,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
    )


def compile_func(
    mdl_key: Tuple[Set[PropPos], bool],
    temp_folder: Path,
    mdl_name: str,
    lookup_model: Callable[[str], Tuple[QC, Model]],
) -> None:
    """Build this merged model."""
    LOGGER.info('Compiling {}...', mdl_name)
    prop_pos, has_coll = mdl_key

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
    coll_mesh = None  #  type: Optional[Mesh]

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
        angles = Angle(prop.pit, prop.yaw, prop.rol)

        ref_mesh.append_model(child_ref, angles, offset, prop.scale * qc.ref_scale)

        if has_coll and child_coll is not None:
            if coll_mesh is None:
                coll_mesh = Mesh.blank('static_prop')
            coll_mesh.append_model(child_coll, angles, offset, prop.scale * qc.phy_scale)

    with (temp_folder / 'reference.smd').open('wb') as fb:
        ref_mesh.export(fb)

    # Generate  a  blank animation.
    with (temp_folder / 'anim.smd').open('wb') as fb:
        Mesh.blank('static_prop').export(fb)

    if coll_mesh is not None:
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

        if coll_mesh is not None:
            f.write(QC_COLL_TEMPLATE)


def build_collision(qc: QC, prop: PropPos, ref_mesh: Mesh) -> Optional[Mesh]:
    """Get the correct collision mesh for this model."""
    if prop.solidity is CollType.NONE:  # Non-solid
        return None
    elif prop.solidity is CollType.VPHYS or prop.solidity is CollType.BSP:
        if qc.phy_smd is None:
            return None
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

            _coll_cache[qc.phy_smd] = coll
            return coll
    # Else, it's one of the three bounding box types.
    # We don't really care about which.
    bbox_min, bbox_max = Vec.bbox(
        vert.pos
        for tri in
        ref_mesh.triangles
        for vert in tri
    )
    return Mesh.build_bbox('static_prop', 'phy', bbox_min, bbox_max)


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
                    if tok.peek()[0] is Token.BRACE_OPEN:
                        for body_type, body_value in tok.block('$collisionmodel'):
                            if body_type is Token.STRING and body_value.casefold() == '$concave':
                                is_concave = True

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
                cache_props = Property.parse(f).find_key('qc', [])
            # Added later, remake if not present.
            if 'concave' not in cache_props:
                raise KeyValError
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
    with TemporaryDirectory() as tempdir, fsys:
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
    else:
        cache_props['ref'] = ''  # Mark as not present.

    with info_path.open('w') as f:
        for line in cache_props.export():
            f.write(line)
    return qc


def group_props_ent(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    rejected: List[StaticProp],
    get_model: Callable[[str], Tuple[Optional[QC], Optional[Model]]],
    bbox_ents: List[Entity],
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, merge props according to the provided ents."""
    # Ents with group names. We have to split those by filter too.
    grouped_sets: Dict[Tuple[str, FrozenSet[str]], CombineVolume] = {}
    # Skinset filter -> volumes that match.
    sets_by_skin: Dict[FrozenSet[str], List[CombineVolume]] = defaultdict(list)

    empty_fs = frozenset('')

    for ent in bbox_ents:
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

        # Compute 6 planes to use for collision detection.
        mat = Matrix.from_angle(Angle.from_str(ent['angles']))
        mins, maxes = Vec.bbox(
            Vec.from_str(ent['mins']),
            Vec.from_str(ent['maxs']),
        )
        size = maxes - mins
        # Enlarge slightly to ensure it never has a zero area.
        # Otherwise the normal could potentially be invalid.
        mins -= 0.05
        maxes += 0.05

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

        combine_set.volume += size.x * size.y * size.z
        # For each direction, compute a position on the plane and
        # the normal vector.
        combine_set.collision.append([
            (
                origin + Vec.with_axes(axis, offset) @ mat,
                Vec.with_axes(axis, norm) @ mat,
            )
            for offset, norm in zip([mins, maxes], (-1, 1))
            for axis in ('x', 'y', 'z')
        ])

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
            rejected.extend(group)
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

    # Finally, reject all the ones not in a bbox.
    for group in prop_groups.values():
        rejected.extend(group)
    # And log unused groups
    for combine_set_list in sets_by_skin.values():
        for combine_set in combine_set_list:
            if not combine_set.used:
                LOGGER.warning('Unused comp_propcombine_set {}', combine_set.desc)


def group_props_auto(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    rejected: List[StaticProp],
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
            rejected.extend(group)
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
                rejected.append(center)
                continue

            bbox_min, bbox_max = Vec.bbox(prop.origin for prop in cluster)
            center_pos = (bbox_min + bbox_max) / 2

            cluster_list = []

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
            else:
                rejected.extend(selected_props)


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
    auto_range: float=0,
    min_cluster: int=2,
    debug_tint: bool=False,
    debug_dump: bool=False,
) -> None:
    """Combine props in this map."""

    # First parse out the bbox ents, so they are always removed.
    bbox_ents = list(bsp_ents.by_class['comp_propcombine_set'])
    for ent in bbox_ents:
        ent.remove()

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot propcombine!')
        return

    if not qc_folders and decomp_cache_loc is None:
        # If gameinfo is blah/game/hl2/gameinfo.txt,
        # QCs should be in blah/content/ according to Valve's scheme.
        # But allow users to override this.
        # If Crowbar's path is provided, that means they may want to just supply nothing.
        qc_folders = [game.path.parent.parent / 'content']

    # Parse through all the QC files.
    LOGGER.info('Parsing QC files. Paths: \n{}', '\n'.join(map(str, qc_folders)))
    qc_map: Dict[str, Optional[QC]] = {}
    for qc_folder in qc_folders:
        load_qcs(qc_map, qc_folder)
    LOGGER.info('Done! {} props.', len(qc_map))

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
    prop_groups = defaultdict(list)  # type: Dict[Optional[tuple], List[StaticProp]]

    # This holds the list of all props we want in the map -
    # combined ones, and any we reject for whatever reason.
    final_props: List[StaticProp] = []
    rejected: List[StaticProp] = []

    if bbox_ents:
        LOGGER.info('Propcombine sets present ({}), combining...', len(bbox_ents))
        grouper = group_props_ent(
            prop_groups, rejected,
            get_model,
            bbox_ents,
            min_cluster,
        )
    elif auto_range > 0:
        LOGGER.info('Automatically finding propcombine sets...')
        grouper = group_props_auto(
            prop_groups, rejected,
            auto_range,
            min_cluster,
        )
    else:
        # No way provided to choose props.
        LOGGER.info('No propcombine groups provided.')
        return

    for prop in bsp.static_props():
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
    
    group_count = 0
    with ModelCompiler(
        game,
        studiomdl_loc,
        pack,
        map_name,
        'propcombine',
    ) as compiler:
        for group in grouper:
            grouped_prop = combine_group(compiler, group, get_model)
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
        'Combined {} props into {}:\n - {} grouped models\n - {} ineligable\n - {} had no group',
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

    bsp.write_static_props(final_props)
