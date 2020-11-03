"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import os
import random
import colorsys
import functools
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    Optional, Tuple, Callable, NamedTuple,
    FrozenSet, Dict, List, Set,
    Iterator,
)

from srctools import Vec, VMF, Entity, conv_int
from srctools.tokenizer import Tokenizer, Token
from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp
from srctools.mdl import Model
from srctools.smd import Mesh
from srctools.compiler.mdl_compiler import ModelCompiler


LOGGER = get_logger(__name__)

QC = NamedTuple('QC', [
    ('path', str),  # QC path.
    ('ref_smd', str),  # Location of main visible geometry.
    ('phy_smd', Optional[str]),  # Relative location of collision model, or None
    ('ref_scale', float),  # Scale of main model.
    ('phy_scale', float),  # Scale of collision model.
])

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

class DynamicModel(Exception):
    """Used as flow control."""


def unify_mdl(path: str):
    """Compute a 'canonical' path for a given model."""
    path = path.casefold().replace('\\', '/')
    if not path.startswith('models/'):
        path = 'models/' + path
    if not path.endswith('.mdl'):
        path = path + '.mdl'
    return path


def bsp_collision(point: Vec, planes: List[Tuple[Vec, Vec]]) -> bool:
    """Check if the given position is inside a BSP node."""
    for pos, norm in planes:
        off = pos - point
        # This is the actual distance, so we'll use a rather large
        # "epsilon" to catch objects close to the edges.
        if Vec.dot(off, norm) < -0.1:
            return False
    return True


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
        avg_yaw += prop.angles.y
        visleafs.update(prop.visleafs)

    avg_pos /= len(props)

    prop_pos = set()
    for prop in props:
        origin = round((prop.origin - avg_pos), 7)
        angles = round(Vec(prop.angles), 7)
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
    mdl_name = compiler.get_model(
        (frozenset(prop_pos), has_coll),
        functools.partial(compile_func, lookup_model),
    )

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model=mdl_name,
        origin=avg_pos,
        angles=Vec(0, 270, 0),
        scaling=1.0,
        visleafs=sorted(visleafs),
        solidity=(CollType.VPHYS if has_coll else CollType.NONE).value,
        flags=props[0].flags,
        lighting_origin=avg_pos,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
    )


def compile_func(
    lookup_model: Callable[[str], Tuple[QC, Model]],
    mdl_key: Tuple[Set[PropPos], bool],
    temp_folder: Path,
    mdl_name: str,
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
            for tri in child_ref.triangles:
                for vert in tri:
                    vert.pos.rotate(0, 90, 0, round_vals=False)
                    vert.norm.rotate(0, 90, 0, round_vals=False)

            _mesh_cache[qc, prop.skin] = child_ref

        child_coll = build_collision(qc, prop, child_ref)

        offset = Vec(prop.x, prop.y, prop.z)
        angles = Vec(prop.pit, prop.yaw, prop.rol)

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

            for tri in coll.triangles:
                for vert in tri:
                    vert.pos.rotate(0, 90, 0, round_vals=False)
                    vert.norm.rotate(0, 90, 0, round_vals=False)

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
                model_name,
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
            )


def parse_qc(qc_loc: Path, qc_path: Path) -> Optional[Tuple[
    str,
    float, Path,
    float, Optional[Path]
]]:
    """Parse a single QC file."""
    model_name = ref_smd = phy_smd = None
    scale_factor = ref_scale = phy_scale = 1.0

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
        model_name,
        ref_scale, ref_smd,
        phy_scale, phy_smd,
    )


def group_props_ent(
    prop_groups: Dict[Optional[tuple], List[StaticProp]],
    rejected: List[StaticProp],
    get_model: Callable[[str], Tuple[Optional[QC], Optional[Model]]],
    bbox_ents: List[Entity],
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, merge props according to the provided ents."""
    # (name, skinset) -> list of boxes, constructed as 6 (pos, norm) tuples.
    combine_sets = defaultdict(list)  # type: Dict[Tuple[str, FrozenSet[str]], List[List[Tuple[Vec, Vec]]]]

    empty_fs = frozenset('')

    for ent in bbox_ents:
        # Either provided name, or unique value.
        name = ent['name'] or format(int(ent['hammerid']), 'X')
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
        angles = Vec.from_str(ent['angles'])
        mins, maxes = Vec.bbox(
            Vec.from_str(ent['mins']),
            Vec.from_str(ent['maxs']),
        )
        # Enlarge slightly to ensure it never has a zero area.
        # Otherwise the normal could potentially be invalid.
        mins -= 0.05
        maxes += 0.05

        # For each direction, compute a position on the plane and
        # the normal vector.
        combine_sets[name, skinset].append([
            (
                origin + Vec.with_axes(axis, offset).rotate(*angles),
                Vec.with_axes(axis, norm).rotate(*angles),
            )
            for offset, norm in zip([mins, maxes], (-1, 1))
            for axis in ('x', 'y', 'z')
        ])

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

        for (name, skinset), boxes in combine_sets.items():
            if skinset and skinset != group_skinset:
                continue  # No match
            found = defaultdict(list)  # type: Dict[int, List[StaticProp]]
            for prop in list(group):
                for box in boxes:
                    if bsp_collision(prop.origin, box):
                        # Group by this box object's identity.
                        # That's a cheap way to keep each propcombine set
                        # grouped uniquely.
                        found[id(boxes)].append(prop)
                        break

            for subgroup in found.values():
                actual = set(subgroup).intersection(group)
                if len(actual) >= min_cluster:
                    yield list(actual)
                    for prop in actual:
                        group.remove(prop)

    # Finally, reject all the ones not in a bbox.
    for group in prop_groups.values():
        rejected.extend(group)


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

            cluster_list.sort(key=lambda t: t[1])
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
    studiomdl_loc: Path=None,
    qc_folders: List[Path]=None,
    auto_range: float=0,
    min_cluster: int=2,
    debug_tint: bool=False,
) -> None:
    """Combine props in this map."""

    # First parse out the bbox ents, so they are always removed.
    bbox_ents = list(bsp_ents.by_class['comp_propcombine_set'])
    for ent in bbox_ents:
        ent.remove()

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot propcombine!')
        return

    if not qc_folders:
        # If gameinfo is blah/game/hl2/gameinfo.txt,
        # QCs should be in blah/content/ according to Valve's scheme.
        # But allow users to override this.
        qc_folders = [game.path.parent.parent / 'content']

    # Parse through all the QC files.
    LOGGER.info('Parsing QC files. Paths: \n{}', '\n'.join(map(str, qc_folders)))
    qc_map = {}  # type: Dict[str, QC]
    for qc_folder in qc_folders:
        load_qcs(qc_map, qc_folder)
    LOGGER.info('Done! {} props.', len(qc_map))

    map_name = Path(bsp.filename).stem

    # Don't re-parse models continually.
    mdl_map = {}  # type: Dict[str, Model]
    # Wipe these, if they're being used again.
    _mesh_cache.clear()
    _coll_cache.clear()

    def get_model(filename: str) -> Tuple[Optional[QC], Optional[Model]]:
        """Given a filename, load/parse the QC and MDL data."""
        key = unify_mdl(filename)
        try:
            qc = qc_map[key]
        except KeyError:
            return None, None
        try:
            model = mdl_map[key]
        except KeyError:
            try:
                mdl_file = pack.fsys[filename]
            except FileNotFoundError:
                # We don't have this model, we can't combine...
                return None, None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)
        return qc, model

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
            prop.flags.value,
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

    if bbox_ents:
        LOGGER.info('Propcombine sets present ({}), combining...', len(bbox_ents))
        grouper = group_props_ent(
            prop_groups, final_props,
            get_model,
            bbox_ents,
            min_cluster,
        )
    elif auto_range > 0:
        LOGGER.info('Automatically finding propcombine sets...')
        grouper = group_props_auto(
            prop_groups, final_props,
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
    final_props.extend(prop_groups.pop(None, []))

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
                grouped_prop.tint = round(Vec(*colorsys.hsv_to_rgb(random.random(), 1, 1)) * 255)
            final_props.append(grouped_prop)

    LOGGER.info(
        'Combined {} props to {} props using {} groups.',
        prop_count,
        len(final_props),
        compiler.model_folder,
    )
    # If present, delete old cache file. We'll have cleaned up the models.
    try:
        os.remove(compiler.model_folder_abs / 'cache.vdf')
    except FileNotFoundError:
        pass

    bsp.write_static_props(final_props)
