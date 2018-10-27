"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Iterator, Optional

from srctools import Vec
from srctools.tokenizer import Tokenizer, Token

from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, BSP_LUMPS, StaticProp, StaticPropFlags
from srctools.mdl import Model
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
$cdmaterials {cdmats}

$body body blank.smd

$sequence idle blank act_idle 1
'''

QC_COLL_TEMPLATE = '''
$collisionmodel "blank" {
    $maxconvexpieces 2048
    $concaveperjoint
    $automass
    $remove2d
    $concave
'''

# No bones, no animation, no geometry...
BLANK_SMD  = b'''\
version 1
nodes
  0 "static_prop" -1
end
skeleton
  time 0
    0 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
end
triangles
end
'''

MDL_EXTS = [
    '.mdl',
    '.phy',
    '.dx90.vtx',
    '.dx80.vtx',
    '.sw.vtx',
    '.vvd',
]


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


def load_qcs(game: Game) -> Dict[str, QC]:
    """Parse through all the QC files to match to compiled models."""
    # If gameinfo is blah/game/hl2/gameinfo.txt,
    # QCs should be in blah/content/....

    qc_map = {}

    content_path = game.path.parent.parent / 'content'
    for qc_path in content_path.rglob('*.qc'):  # type: Path
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
                        elif token_value == "$bodygroup":
                            tok.expect(Token.STRING)  # group name.
                            tok.expect(Token.BRACE_OPEN)
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
                            '$lod',
                            '$jigglebone',
                            '$keyvalues',
                        ):
                            raise DynamicModel

        except DynamicModel:
            # It's a dynamic QC, we can't combine.
            continue
        if model_name is None or ref_smd is None:
            # Malformed...
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
    game: Game,
    pack: PackList,
    map_name: str,
    props: List[StaticProp],
    counter: int,
) -> StaticProp:
    """Given a set of props, merge them together to produce a new prop."""

    bbox_min, bbox_max = Vec.bbox(prop.origin for prop in props)

    center_pos = (bbox_min + bbox_max) / 2

    # Unify these properties.
    surfprops = set()
    cdmats = set()
    visleafs = set()

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

    with TemporaryDirectory(prefix='autocomb_') as temp_dir:
        with open(temp_dir + '/blank.smd', 'wb') as f:
            f.write(BLANK_SMD)
        with open(temp_dir + '/model.qc', 'w') as f:
            f.write(QC_TEMPLATE.format(
                path=prop_name,
                surf=surfprop,
                cdmats=' '.join('"{}"'.format(mat) for mat in sorted(cdmats))
            ))

            has_coll = False

            for prop in props:
                qc = qc_map[unify_mdl(prop.model)]
                f.write(
                    '$appendsource "{}" "offset '
                    'pos[ {:.3f} {:.3f} {:.3f} ] '
                    'angle[ {:.3f} {:.3f} {:.3f} ] '
                    'scale[ {:.3f} '
                    ']"\n'.format(
                        qc.ref_smd,
                        prop.origin.x - center_pos.x,
                        prop.origin.y - center_pos.y,
                        prop.origin.z - center_pos.z,
                        prop.angles.x,
                        prop.angles.y,
                        prop.angles.z,
                        qc.ref_scale
                    )
                )
                if qc.phy_smd:
                    has_coll = True

            if has_coll:
                f.write(QC_COLL_TEMPLATE)
                for prop in props:
                    qc = qc_map[unify_mdl(prop.model)]
                    if qc.phy_smd:
                        f.write(
                            '    $addconvexsrc "{}" "offset '
                            'pos[ {:.3f} {:.3f} {:.3f} ] '
                            'angle[ {:.3f} {:.3f} {:.3f} ] '
                            'scale[ {:.3f} '
                            ']"\n'.format(
                                qc.phy_smd,
                                prop.origin.x - center_pos.x,
                                prop.origin.y - center_pos.y,
                                prop.origin.z - center_pos.z,
                                prop.angles.x,
                                prop.angles.y,
                                prop.angles.z,
                                qc.phy_scale
                            )
                        )
                f.write('}\n')
        args = [
            'F:/Git/Desolation/game/bin/win32/studiomdl.exe',
            '-nop4',
            '-game', str(game.path), temp_dir + '/model.qc',
        ]
        subprocess.run(args)

    full_model_path = game.path / 'models' / prop_name
    for ext in MDL_EXTS:
        try:
            with open(str(full_model_path) + ext, 'rb') as f:
                pack.pack_file(
                    'models/{}{}'.format(prop_name, ext),
                    data=f.read(),
                )
        except FileNotFoundError:
            pass

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model='models/{}.mdl'.format(prop_name),
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
        min_dx_level=0,
        max_dx_level=0,
        min_cpu_level=0,
        max_cpu_level=0,
        min_gpu_level=0,
        max_gpu_level=0,
        tint=props[0].tint,
        renderfx=props[0].renderfx,
        disable_on_xbox=False,
    )


def combine(
    bsp: BSP,
    pack: PackList,
    game: Game,
):
    """Combine props in this map."""
    # Parse through all the QC files.
    LOGGER.info('Parsing QC files...')
    qc_map = load_qcs(game)
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
            model.flags,
            prop.flags,
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

    # Don't worry about distance.

    final_props = []
    for ind, (key, props) in enumerate(prop_groups.items()):
        # First remove props that aren't actually mergeable.
        if key is None:
            final_props.extend(props)
            continue
        for prop in props[:]:
            if unify_mdl(prop.model) not in qc_map:
                final_props.append(prop)
                props.remove(prop)

        if len(props) < 2:
            # No point combining that.
            final_props.extend(props)
            continue

        final_props.append(merge_props(
            qc_map,
            mdl_map,
            game,
            pack,
            map_name,
            props,
            ind,
        ))

    LOGGER.info('Combined {} props to {} props', prop_count, len(final_props))

    bsp.write_static_props(final_props)
