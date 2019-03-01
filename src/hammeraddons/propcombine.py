"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Set, Optional, Tuple

from srctools import Vec, partition
from srctools.tokenizer import Tokenizer, Token

from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp, StaticPropFlags
from srctools.mdl import Model
from srctools.smd import Mesh
from collections import defaultdict, namedtuple


LOGGER = get_logger(__name__)

QC = namedtuple('QC', [
    'path',  # QC path.
    'ref_smd',    # Location of main visible geometry.
    'phy_smd',    # Relative location of collision model, or None
    'ref_scale',  # Scale of main model.
    'phy_scale',  # Scale of collision model.
])

QC_TEMPLATE = '''\
$staticprop
$modelname "{path}"
$surfaceprop "{surf}"

$body body "{ref_mesh}"

$sequence idle anim act_idle 1
'''

QC_COLL_TEMPLATE = '''
$collisionmodel "{}" {{
    $maxconvexpieces 2048
    $automass
    $concave
}}
'''


MDL_EXTS = [
    '.mdl',
    '.phy',
    '.dx90.vtx',
    '.dx80.vtx',
    '.sw.vtx',
    '.vvd',
]

MAX_GROUP = 24  # Studiomdl does't allow more than this...


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


def load_qcs(qc_folder: Path) -> Dict[str, QC]:
    """Parse through all the QC files to match to compiled models."""
    qc_map = {}

    for qc_path in qc_folder.rglob('*.qc'):  # type: Path
        model_name = ref_smd = phy_smd = None
        scale_factor = ref_scale = phy_scale = 1.0
        qc_loc = qc_path.parent
        try:
            with open(qc_path) as f:
                tok = Tokenizer(f, qc_path, allow_escapes=False)
                for token_type, token_value in tok:

                    if model_name and ref_smd and phy_smd:
                        break

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
                                    raise DynamicModel
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
                                            raise DynamicModel
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
                            '$keyvalues',
                            # Allow LOD models, propcombine is better than that.
                            # '$lod',
                        ):
                            raise DynamicModel

        except DynamicModel:
            # It's a dynamic QC, we can't combine.
            continue
        if model_name is None or ref_smd is None:
            # Malformed...
            LOGGER.warning('Cannot parse "{}"... ({}, {})', qc_path, model_name, ref_smd)
            continue

        # We can't parse FBX files right now.
        if ref_smd.suffix.casefold() not in ('.smd', '.dmx'):
            LOGGER.warning('Reference mesh not a SMD/DMX:\n{}', ref_smd)
            continue

        if phy_smd is not None:
            if phy_smd.suffix.casefold() not in ('.smd', '.dmx'):
                LOGGER.warning('Collision mesh not a SMD/DMX:\n{}', ref_smd)
                continue

        qc_map[unify_mdl(model_name)] = QC(
            str(qc_path).replace('\\', '/'),
            str(ref_smd).replace('\\', '/'),
            str(phy_smd).replace('\\', '/') if phy_smd else None,
            ref_scale,
            phy_scale,
        )
    return qc_map


def merge_props(
    qc_map: Dict[str, QC],
    mdl_map: Dict[str, Model],
    mesh_cache: Dict[Tuple[QC, int], Tuple[Mesh, Optional[Mesh]]],
    temp_folder: Path,
    game: Game,
    studiomdl_loc: Path,
    pack: PackList,
    map_name: str,
    props: List[StaticProp],
    counter: int,
) -> StaticProp:
    """Given a set of props, merge them together to produce a new prop."""

    bbox_min, bbox_max = Vec.bbox(prop.origin for prop in props)

    center_pos = (bbox_min + bbox_max) / 2

    # Unify these properties.
    surfprops = set()  # type: Set[str]
    cdmats = set()  # type: Set[str]
    visleafs = set()  # type: Set[int]

    for prop in props:
        mdl = mdl_map[unify_mdl(prop.model)]
        surfprops.add(mdl.surfaceprop.casefold())
        cdmats.update(mdl.cdmaterials)
        visleafs.update(prop.visleafs)

    if len(surfprops) > 1:
        raise ValueError('Multiple surfaceprops? Should be filtered out.')

    [surfprop] = surfprops

    if counter > 0xFFFF:
        raise ValueError('More than 65K models, how??')

    prop_name = 'maps/{}/propcombine/merge_{:04X}'.format(map_name, counter)

    ref_mesh = Mesh.blank('static_prop')
    coll_mesh = None  #  type: Optional[Mesh]

    for prop in props:
        qc = qc_map[unify_mdl(prop.model)]
        mdl = mdl_map[unify_mdl(prop.model)]
        try:
            child_ref, child_coll = mesh_cache[qc, prop.skin]
        except KeyError:
            LOGGER.info('Parsing ref "{}"', qc.ref_smd)
            with open(qc.ref_smd, 'rb') as fb:
                child_ref = Mesh.parse_smd(fb)
            if qc.phy_smd is not None:
                LOGGER.info('Parsing coll "{}"', qc.phy_smd)
                with open(qc.phy_smd, 'rb') as fb:
                    child_coll = Mesh.parse_smd(fb)
            else:
                child_coll = None

            if prop.skin != 0 and prop.skin <= len(mdl.skins):
                # We need to rename the materials to match the skin.
                swap_skins = dict(zip(
                    mdl.skins[0],
                    mdl.skins[prop.skin]
                ))
                for tri in child_ref.triangles:
                    tri.mat = swap_skins.get(tri.mat, tri.mat)

            # For some reason all the SMDs are rotated badly...
            for tri in child_ref.triangles:
                for vert in tri:
                    vert.pos.rotate(0, 90, 0, round_vals=False)
                    vert.norm.rotate(0, 90, 0, round_vals=False)
            if child_coll is not None:
                for tri in child_coll.triangles:
                    for vert in tri:
                        vert.pos.rotate(0, 90, 0, round_vals=False)
                        vert.norm.rotate(0, 90, 0, round_vals=False)

            mesh_cache[qc, prop.skin] = child_ref, child_coll

        offset = prop.origin - center_pos

        ref_mesh.append_model(child_ref, prop.angles, offset)

        if child_coll is not None:
            if coll_mesh is None:
                coll_mesh = Mesh.blank('static_prop')
            coll_mesh.append_model(child_coll, prop.angles, offset)

    prefix = str(temp_folder / '{:04X}'.format(counter))

    with open(prefix + '_ref.smd', 'wb') as fb:
        ref_mesh.export(fb)

    if coll_mesh is not None:
        with open(prefix + '_phy.smd', 'wb') as fb:
            coll_mesh.export(fb)

    with open(prefix + '.qc', 'w') as f:
        f.write(QC_TEMPLATE.format(
            path=prop_name,
            surf=surfprop,
            ref_mesh=prefix + '_ref.smd',
        ))

        for mat in sorted(cdmats):
            f.write('$cdmaterials "{}"\n'.format(mat))

        if coll_mesh is not None:
            f.write(QC_COLL_TEMPLATE.format(prefix + '_phy.smd'))

    args = [
        str(studiomdl_loc.resolve()),
        '-nop4',
        '-game', str(game.path),
        prefix + '.qc',
    ]
    subprocess.run(args)

    full_model_path = game.path / 'models' / prop_name
    for ext in MDL_EXTS:
        try:
            with open(str(full_model_path) + ext, 'rb') as fb:
                pack.pack_file(
                    'models/{}{}'.format(prop_name, ext),
                    data=fb.read(),
                )
        except FileNotFoundError:
            pass

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model='models/{}.mdl'.format(prop_name),
        prop_id=props[0].id,  # We're replacing this ID so we know it's free.
        origin=center_pos,
        angles=Vec(0, 270, 0),
        scaling=1.0,
        visleafs=sorted(visleafs),
        solidity=props[0].solidity,
        flags=props[0].flags,
        skin=0,
        min_fade=-1,
        max_fade=-1,
        lighting_origin=center_pos,
        fade_scale=1.0,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
        disable_on_xbox=False,
    )


def group_props(
    prop_groups: Dict[object, List[StaticProp]],
    rejected: List[StaticProp],
    dist,
    min_cluster,
):
    """Given the groups of props, find close props to merge."""
    # Each of these groups cannot be merged with other ones.

    # These are models we cannot merge no matter what -
    # no source files etc.
    if None in prop_groups:
        rejected.extend(prop_groups.pop(None))

    dist_sq = dist * dist
    large_dist_sq = 4 * dist_sq

    for key, group in prop_groups.items():
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

            cluster_list = [
                prop for prop in cluster
                if (center_pos - prop.origin).mag_sq() <= dist_sq
            ]

            if len(cluster_list) >= min_cluster:
                todo.difference_update(cluster_list)
                for part in partition(cluster_list, MAX_GROUP):
                    yield part
            else:
                rejected.append(center)


def combine(
    bsp: BSP,
    pack: PackList,
    game: Game,
    studiomdl_loc: Path=None,
    qc_folder: Path=None,
):
    """Combine props in this map."""
    if studiomdl_loc is None:
        studiomdl_loc = game.bin_folder() / 'studiomdl.exe'

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot propcombine!')
        return

    if qc_folder is None:
        # If gameinfo is blah/game/hl2/gameinfo.txt,
        # QCs should be in blah/content/ according to Valve's scheme.
        # But allow users to override this.
        qc_folder = game.path.parent.parent / 'content'

    # Parse through all the QC files.
    LOGGER.info('Parsing QC files. Path: {}', qc_folder)
    qc_map = load_qcs(qc_folder)
    LOGGER.info('Done! {} props.', len(qc_map))

    map_name = Path(bsp.filename).stem

    # Don't re-parse models continually.
    mdl_map = {}  # type: Dict[str, Model]

    def get_grouping_key(prop: StaticProp) -> object:
        """Compute a grouping key for this prop.

        Only props with matching key can be possibly combined.
        If None it cannot be combined.
        """
        # If a lighting origin was specified, that overrides the default.
        if prop.flags & StaticPropFlags.HAS_LIGHTING_ORIGIN:
            return None

        key = unify_mdl(prop.model)

        if key not in qc_map:
            # We don't have source for this...
            return None

        try:
            model = mdl_map[key]
        except KeyError:
            try:
                mdl_file = pack.fsys[prop.model]
            except FileNotFoundError:
                # We don't have this model, we can't combine...
                return None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)

        try:
            skinset = model.skins[prop.skin]
        except IndexError:
            skinset = model.skins[0]

        return (
            model.flags.value,
            prop.flags.value,
            model.contents,
            model.surfaceprop,
            prop.solidity,
            prop.renderfx,
            *prop.tint,
            *model.skins[0],
            *skinset,
        )

    prop_count = 0

    # First, construct groups of props that can possibly be combined.
    prop_groups = defaultdict(list)  # type: Dict[object, List[StaticProp]]
    for prop in bsp.static_props():
        prop_groups[get_grouping_key(prop)].append(prop)
        prop_count += 1

    # This holds the list of all props we want in the map -
    # combined ones, and any we reject for whatever reason.
    final_props = []  # type: List[StaticProp]

    with TemporaryDirectory(prefix='autocomb_') as temp_dir:
        mesh_cache = {}
        temp_path = Path(temp_dir)

        with open(temp_dir + '/anim.smd', 'wb') as f:
            Mesh.blank('static_prop').export(f)

        for ind, group in enumerate(group_props(prop_groups, final_props, 128, 2)):
            final_props.append(merge_props(
                qc_map,
                mdl_map,
                mesh_cache,
                temp_path,
                game,
                studiomdl_loc,
                pack,
                map_name,
                group,
                ind,
            ))

    LOGGER.info('Combined {} props to {} props', prop_count, len(final_props))

    bsp.write_static_props(final_props)
