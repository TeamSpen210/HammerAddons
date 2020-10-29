"""Manages potential models that are being generated.

Each comes with a key, used to identify a previously compiled version.
We can then reuse already compiled versions.
"""
import os
import pickle
from typing import Dict, Callable, TypeVar, NamedTuple, Tuple, Set, List
from pathlib import Path

from srctools import KeyValError, Property
from srctools.game import Game
from srctools.packlist import PackList, LOGGER


ModelKey = TypeVar('ModelKey')

class ModelCompiler:
    """Manages the set of merged models that have been generated."""
    def __init__(
        self,
        temp_folder: Path,
        game: Game,
        studiomdl_loc: Path,
        pack: PackList,
        map_name: str,
        folder_name: str,
    ) -> None:
        # The models already constructed: key -> model name.
        self.built_models: Dict[ModelKey, str] = {}

        # The random indexes we use to produce filenames.
        self._mdl_names: Set[str] = set()

        # The stuff we need to know to build models.
        self.game = game
        self.studiomdl_loc = str(studiomdl_loc.resolve())
        self.model_folder = 'maps/{}/{}/'.format(map_name, folder_name)
        self.model_folder_abs = game.path / self.model_folder
        self.temp_folder = temp_folder
        self.pack = pack

        # Ensure the folder exists.
        os.makedirs(self.model_folder, exist_ok=True)

    def load(self) -> None:
        """Load from the cache file."""
        try:
            with (self.model_folder_abs / 'manifest.bin').open('rb') as f:
                data: List[ModelKey] = pickle.load(f)
                assert isinstance(data, list)
        except FileNotFoundError:
            return
        except Exception:
            LOGGER.warning('Could not parse existing models file:', .exc_info=True)
            return


        for model_key in data:
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
