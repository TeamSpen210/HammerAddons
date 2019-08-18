"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import os
import subprocess
import random
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Optional, Tuple, Callable, NamedTuple,
    FrozenSet, Dict, List, Set,
    Iterator,
)

from srctools import (
    Vec, VMF, Entity, conv_int, Property, AtomicWriter, KeyValError,
)
from srctools.tokenizer import Tokenizer, Token

from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp, StaticPropFlags
from srctools.mdl import Model
from srctools.smd import Mesh
from collections import defaultdict


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
$modelname "{path}.mdl"
$surfaceprop "{surf}"

$body body "{ref_mesh}"

$contents {contents}

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


def in_bbox(pos: Vec, bb_min: Vec, bb_max: Vec) -> bool:
    """Check if the given position is inside a bounding box."""
    if pos.x < bb_min.x or bb_max.x < pos.x:
        return False
    if pos.y < bb_min.y or bb_max.y < pos.y:
        return False
    if pos.z < bb_min.z or bb_max.z < pos.z:
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


# 'key' used to match models to each other.
PropPos = NamedTuple('PropPos', [
    ('x', float), ('y', float), ('z', float),
    ('pit', float), ('yaw', float), ('rol', float),
    ('model', str),
    ('skin', int),
    ('scale', float),
    ('solidity', CollType),
])


class MergedModel:
    """Represents a model that has been merged together."""
    def __init__(self, name: str, has_coll: bool) -> None:
        self.name = name
        self.has_coll = has_coll
        self.used = False


class ModelManager:
    """Manages the set of merged models that have been generated."""

    def __init__(
        self,
        temp_folder: Path,
        game: Game,
        studiomdl_loc: Path,
        pack: PackList,
        qc_map: Dict[str, QC],
        map_name: str,
        get_model: Callable[[str], Optional[Model]],
    ) -> None:
        # The models already constructed.
        self._mdl_cache = {}  # type: Dict[FrozenSet[PropPos], MergedModel]

        # Cache of the SMD models we have already parsed, so we don't need
        # to parse them again. The second is the collision model.
        self._mesh_cache = {}  # type: Dict[Tuple[QC, int], Mesh]
        self._coll_cache = {}  # type: Dict[str, Mesh]

        # The random indexes we use to produce filenames.
        self._mdl_names = set()  # type: Set[str]

        # Function to load and parse a MDL.
        self.lookup_model = get_model
        # The QCs we have parsed.
        self.qc_map = qc_map

        # The stuff we need to know to build models.
        self.game = game
        self.studiomdl_loc = str(studiomdl_loc.resolve())
        self.model_folder = 'maps/{}/propcombine/'.format(map_name)
        self.temp_folder = temp_folder
        self.pack = pack

        os.makedirs(self.model_folder, exist_ok=True)

    def load(self) -> None:
        """Load from the cache file."""
        with open(str(self.game.path / 'models' / self.model_folder / 'cache.vdf')) as f:
            prop_block = Property.parse(f)
        for prop in prop_block:
            pos_set = set()
            for pos_block in prop.find_all('model'):
                origin = round(pos_block.vec('origin'), 7)
                angles = round(pos_block.vec('angles'), 7)
                pos_set.add(PropPos(
                    origin.x, origin.y, origin.z,
                    angles.x, angles.y, angles.z,
                    pos_block['filename'],
                    pos_block.int('skin'),
                    pos_block.float('scale', 1.0),
                    CollType(pos_block.int('solid', CollType.VPHYS.value)),
                ))
            mdl_name = prop['name']
            self._mdl_names.add(mdl_name)
            self._mdl_cache[frozenset(pos_set)] = MergedModel(
                mdl_name,
                any(pos.solidity is not CollType.NONE for pos in pos_set),
            )
        LOGGER.info('Found {} existing grouped models.', len(self._mdl_cache))

    def finalise(self) -> None:
        """Write the constructed models to the cache file and remove unused models."""
        used_models = set()

        with AtomicWriter(
            str(self.game.path / 'models' / self.model_folder / 'cache.vdf'),
        ) as f:
            for positions, mdl in self._mdl_cache.items():
                if not mdl.used:
                    continue

                used_models.add(mdl.name)

                prop = Property('PropGroup', [
                    Property('name', mdl.name)
                ])
                for pos in positions:
                    prop.append(Property('Model', [
                        Property('filename', pos.model),
                        Property('skin', str(pos.skin)),
                        Property(
                            'origin',
                            '{} {} {}'.format(pos.x, pos.y, pos.z),
                        ),
                        Property(
                            'angles',
                            '{} {} {}'.format(pos.pit, pos.yaw, pos.rol)
                        ),
                        Property('scale', str(pos.scale)),
                        Property('solid', str(pos.solidity.value)),
                    ]))
                for line in prop.export():
                    f.write(line)

        for mdl_file in (self.game.path / 'models' / self.model_folder).glob('*'):
            if mdl_file.suffix not in {'.mdl', '.phy', '.vtx', '.vvd'}:
                continue

            # Strip all suffixes.
            if mdl_file.name[:mdl_file.name.find('.')] in used_models:
                continue

            LOGGER.info('Culling {}...', mdl_file)
            try:
                mdl_file.unlink()
            except FileNotFoundError:
                pass

    def combine_group(self, props: List[StaticProp]) -> StaticProp:
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
        prop_key = frozenset(prop_pos)

        try:
            merged = self._mdl_cache[prop_key]
        except KeyError:
            # Need to build the model.
            # We don't need to make a collision mesh if the prop is set to
            # not use them.
            has_coll = any(pos.solidity is not CollType.NONE for pos in prop_pos)

            # Figure out a name to use.
            while True:
                mdl_name = 'merge_{:04x}'.format(random.getrandbits(16))
                if mdl_name not in self._mdl_names:
                    self._mdl_names.add(mdl_name)
                    break

            merged = self._mdl_cache[prop_key] = MergedModel(mdl_name, has_coll)
            self.compile(prop_key, merged)

        if not merged.used:
            # Pack it in.
            merged.used = True

            full_model_path = Path(
                self.game.path,
                'models',
                self.model_folder,
                merged.name
            )
            for ext in MDL_EXTS:
                try:
                    with open(str(full_model_path.with_suffix(ext)), 'rb') as fb:
                        self.pack.pack_file(
                            'models/{}{}{}'.format(
                                self.model_folder, merged.name, ext,
                            ),
                            data=fb.read(),
                        )
                except FileNotFoundError:
                    pass

            if not merged.has_coll:
                # Make sure an older collision mesh isn't left behind!
                try:
                    os.remove(full_model_path.with_suffix('.phy'))
                except FileNotFoundError:
                    pass

        # Many of these we require to be the same, so we can read them
        # from any of the component props.
        return StaticProp(
            model='models/{}{}.mdl'.format(self.model_folder, merged.name),
            origin=avg_pos,
            angles=Vec(0, 270, 0),
            scaling=1.0,
            visleafs=sorted(visleafs),
            solidity=CollType.VPHYS.value if merged.has_coll else 0,
            flags=props[0].flags,
            lighting_origin=avg_pos,
            tint=props[0].tint,
            renderfx=props[0].renderfx,
        )

    def compile(
        self,
        prop_pos: FrozenSet[PropPos],
        merged: MergedModel,
    ) -> None:
        """Build this merged model."""
        LOGGER.info('Compiling {}{}.mdl...', self.model_folder, merged.name)

        # Unify these properties.
        surfprops = set()  # type: Set[str]
        cdmats = set()  # type: Set[str]
        contents = set()  # type: Set[int]

        for prop in prop_pos:
            mdl = self.lookup_model(prop.model)
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
            qc = self.qc_map[unify_mdl(prop.model)]
            mdl = self.lookup_model(prop.model)
            assert mdl is not None, prop.model

            try:
                child_ref = self._mesh_cache[qc, prop.skin]
            except KeyError:
                LOGGER.info('Parsing ref "{}"', qc.ref_smd)
                with open(qc.ref_smd, 'rb') as fb:
                    child_ref = Mesh.parse_smd(fb)

                if prop.skin != 0 and prop.skin <= len(mdl.skins):
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

                self._mesh_cache[qc, prop.skin] = child_ref

            child_coll = self._get_collision(qc, prop, child_ref)

            offset = Vec(prop.x, prop.y, prop.z)
            angles = Vec(prop.pit, prop.yaw, prop.rol)

            ref_mesh.append_model(child_ref, angles, offset, prop.scale * qc.ref_scale)

            if merged.has_coll and child_coll is not None:
                if coll_mesh is None:
                    coll_mesh = Mesh.blank('static_prop')
                coll_mesh.append_model(child_coll, angles, offset, prop.scale * qc.phy_scale)

        with open(str(self.temp_folder / (merged.name + '_ref.smd')), 'wb') as fb:
            ref_mesh.export(fb)

        if coll_mesh is not None:
            with open(str(self.temp_folder / (merged.name + '_phy.smd')), 'wb') as fb:
                coll_mesh.export(fb)

        with open(str((self.temp_folder / merged.name).with_suffix('.qc')), 'w') as f:
            f.write(QC_TEMPLATE.format(
                path=self.model_folder + merged.name,
                surf=surfprop,
                ref_mesh=merged.name + '_ref.smd',
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
                f.write(QC_COLL_TEMPLATE.format(merged.name + '_phy.smd'))

        args = [
            self.studiomdl_loc,
            '-nop4',
            '-game', str(self.game.path),
            str((self.temp_folder / merged.name).with_suffix('.qc')),
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL)

    def _get_collision(self, qc: QC, prop: PropPos, ref_mesh: Mesh) -> Optional[Mesh]:
        """Get the correct collision mesh for this model."""
        if prop.solidity is CollType.NONE:  # Non-solid
            return None
        elif prop.solidity is CollType.VPHYS or prop.solidity is CollType.BSP:
            if qc.phy_smd is None:
                return None
            try:
                return self._coll_cache[qc.phy_smd]
            except KeyError:
                LOGGER.info('Parsing coll "{}"', qc.phy_smd)
                with open(qc.phy_smd, 'rb') as fb:
                    coll = Mesh.parse_smd(fb)

                for tri in coll.triangles:
                    for vert in tri:
                        vert.pos.rotate(0, 90, 0, round_vals=False)
                        vert.norm.rotate(0, 90, 0, round_vals=False)

                self._coll_cache[qc.phy_smd] = coll
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

    def use_count(self) -> int:
        """Return the number of used models."""
        return sum(1 for mdl in self._mdl_cache.values() if mdl.used)


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
            if ref_smd.suffix.casefold() not in ('.smd', '.dmx'):
                LOGGER.warning('Reference mesh not a SMD/DMX:\n{}', ref_smd)
                continue

            if phy_smd is not None and phy_smd.suffix.casefold() not in ('.smd', '.dmx'):
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
        tok = Tokenizer(f, qc_path, allow_escapes=False)
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
    get_model: Callable[[str], Optional[Model]],
    bbox_ents: List[Entity],
    min_cluster: int,
) -> Iterator[List[StaticProp]]:
    """Given the groups of props, merge props according to the provided ents."""
    bbox_groups = defaultdict(list)  # type: Dict[Tuple[str, FrozenSet[str]], List[Tuple[Vec, Vec]]]

    empty_fs = frozenset('')

    for ent in bbox_ents:
        # Either provided name, or unique value.
        name = ent['name'] or format(int(ent['hammerid']), 'X')
        origin = Vec.from_str(ent['origin'])

        skinset = empty_fs

        mdl_name = ent['prop']
        if mdl_name:
            mdl = get_model(mdl_name)
            if mdl is not None:
                skinset = frozenset(mdl.iter_textures([conv_int(ent['skin'])]))

        bbox_groups[name, skinset].append(Vec.bbox(
            Vec.from_str(ent['mins']) + origin,
            Vec.from_str(ent['maxs']) + origin,
        ))

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

        for (name, skinset), bboxes in bbox_groups.items():
            if skinset and skinset != group_skinset:
                continue  # No match
            found = defaultdict(list)  # type: Dict[int, List[StaticProp]]
            for prop in list(group):
                for bbox_min, bbox_max in bboxes:
                    if in_bbox(prop.origin, bbox_min, bbox_max):
                        # Group by this bbox list object's identity.
                        found[id(bboxes)].append(prop)
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
) -> None:
    """Combine props in this map."""

    # First parse out the bbox ents, so they are always removed.
    bbox_ents = list(bsp_ents.by_class['comp_propcombine_set'])
    for ent in bbox_ents:
        ent.remove()

    if studiomdl_loc is None:
        studiomdl_loc = game.bin_folder() / 'studiomdl.exe'

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

    def get_model(filename: str) -> Optional[Model]:
        key = unify_mdl(filename)
        try:
            model = mdl_map[key]
        except KeyError:
            try:
                mdl_file = pack.fsys[filename]
            except FileNotFoundError:
                # We don't have this model, we can't combine...
                return None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)
        return model

    def get_grouping_key(prop: StaticProp) -> Optional[tuple]:
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

        model = get_model(prop.model)

        if model is None:
            return None

        return (
            # Must be first!
            frozenset(model.iter_textures([prop.skin])),
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
    final_props = []  # type: List[StaticProp]

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

    with TemporaryDirectory(prefix='autocomb_') as temp_dir:
        # Generate a blank animation.
        with open(temp_dir + '/anim.smd', 'wb') as f:
            Mesh.blank('static_prop').export(f)

        mdl_man = ModelManager(
            Path(temp_dir),
            game,
            studiomdl_loc,
            pack,
            qc_map,
            map_name,
            get_model,
        )

        try:
            mdl_man.load()
        except FileNotFoundError:
            pass  # Make a new one.
        except (IOError, KeyValError):
            LOGGER.warning("Couldn't parse props cache file:", exc_info=True)

        for group in grouper:
            final_props.append(mdl_man.combine_group(group))

    mdl_man.finalise()

    LOGGER.info(
        'Combined {} props to {} props using {} groups.',
        prop_count,
        len(final_props),
        mdl_man.use_count(),
    )

    bsp.write_static_props(final_props)
