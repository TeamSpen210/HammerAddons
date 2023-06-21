"""Handles configuration for the objects appearing inside vactubes."""
from typing import Optional, Tuple, List, Dict
from typing_extensions import TypeAlias
from collections import defaultdict
import os.path
import math

from srctools.packlist import PackList, FileType
from srctools import Entity, Vec, VMF, conv_bool, conv_int
import srctools.logger

from hammeraddons.bsp_transform.packing import make_precache_prop


LOGGER = srctools.logger.get_logger(__name__)

# For prop_weighted_cube, cube type -> model, clean skin, dirty skin
_CUBE_STANDARD = ('models/props/metal_box.mdl', 0, 3)
_CUBE_COMPANION = ('models/props/metal_box.mdl', 1, 1)
_CUBE_REFLECT = ('models/props/reflection_cube.mdl', 0, 1)
_CUBE_SPHERE = ('models/props_gameplay/mp_ball.mdl', 0, 0)
_CUBE_ANTIQUE = ('models/props_underground/underground_weighted_cube.mdl', 0, 0)
CUBE_MODELS_FOR_TYPE = [
    _CUBE_STANDARD, _CUBE_COMPANION,
    _CUBE_REFLECT, _CUBE_SPHERE, _CUBE_ANTIQUE,
]
# The same, but using the old skin property.
CUBE_MODELS_FOR_SKIN = [
    _CUBE_STANDARD, _CUBE_COMPANION,
    _CUBE_STANDARD,  # Standard Activated -> just regular.
    _CUBE_REFLECT, _CUBE_SPHERE, _CUBE_ANTIQUE,
]


class VacObject:
    """An object that can appear in vactubes."""
    def __init__(
        self,
        obj_id: str,
        group: str,
        model_vac: str,
        model_drop: Optional[str],
        offset: Vec,
        weight: int=1,
        skin_tv: int=0,
        skin_drop: int=0,
        skin_vac: int=0,
    ) -> None:
        self.id = obj_id
        self.group = group.casefold().strip()
        self.model_vac = model_vac  # Model for in vactubes
        self.model_drop = model_drop  # If a cube, the real cube model.
        self.weight = weight
        self.offset = offset
        self.skin_tv = skin_tv  # If set, switch scanner TVs to this while passing.
        self.skin_vac = skin_vac
        self.skin_drop = skin_drop

    def __repr__(self) -> str:
        return f'<Vac Object "{os.path.basename(self.model_vac)}">'

    def make_code(self) -> str:
        """Generate the code to construct this object in VScript."""
        if self.model_drop:
            model_code = f'"{self.model_drop}"'
        else:
            model_code = 'null'
        return (
            f'{self.id} <- obj("{self.model_vac}", {self.skin_vac}, '
            f'{model_code}, {self.weight}, "{self.offset}", {self.skin_tv});'
        )


VacObjectDict: TypeAlias = Dict[Tuple[str, str, int], VacObject]


def parse(vmf: VMF, pack: PackList) -> Tuple[int, VacObjectDict, Dict[str, str]]:
    """Parse out the cube objects from the map.

    The return value is the number of objects, a dict of objects, and the
    filenames of the script generated for each group.
    The dict is (group, model, skin) -> object.
    """
    cube_objects: Dict[Tuple[str, str, int], VacObject] = {}
    vac_objects: Dict[str, List[VacObject]] = defaultdict(list)

    for i, ent in enumerate(vmf.by_class['comp_vactube_object']):
        offset = Vec.from_str(ent['origin']) - Vec.from_str(ent['offset'])
        obj = VacObject(
            f'obj_{i:x}',
            ent['group'],
            ent['model'],
            ent['cube_model'],
            offset,
            srctools.conv_int(ent['weight']),
            srctools.conv_int(ent['tv_skin']),
            srctools.conv_int(ent['cube_skin']),
            srctools.conv_int(ent['skin']),
        )
        vac_objects[obj.group].append(obj)
        # Convert the ent into a precache ent, stripping the other keyvalues.
        mdl_name = ent['model']
        ent.clear()
        ent['model'] = mdl_name
        make_precache_prop(ent)
        pack.pack_file(mdl_name, FileType.MODEL, skinset={obj.skin_vac})

        if obj.model_drop:
            cube_objects[
                obj.group,
                obj.model_drop.replace('\\', '/'),
                obj.skin_drop,
            ] = obj

    # Generate and pack the vactube object scripts.
    # Each group is the same, so it can be shared among them all.
    codes = {}
    for group, objects in sorted(vac_objects.items(), key=lambda t: t[0]):
        # First, see if there's a common multiple among the weights, allowing
        # us to simplify.
        multiple = objects[0].weight
        for obj in objects[1:]:
            multiple = math.gcd(multiple, obj.weight)
        if multiple > 1:
            LOGGER.info('Group "{}" has common factor of {}, simplifying.', group, multiple)
        code = []
        for obj in objects:
            obj.weight /= multiple
            code.append(obj.make_code())
        codes[group] = pack.inject_vscript('\n'.join(code))

    return sum(map(len, vac_objects.values())), cube_objects, codes


def find_for_cube(vac_objects: VacObjectDict, group: str, cube: Entity) -> VacObject:
    """Find an object that matches the specified cube entity."""
    potentials: List[Tuple[str, int]] = []
    # Try what's set in the keyvalues first. But if it's a default value, skip so that we use
    # the cube type first.
    model = cube['model'].replace('\\', '/')
    if model not in ('', 'models/props/metal_box.mdl'):
        potentials.append((model, conv_int(cube['skin'])))

    if cube['classname'] == 'prop_weighted_cube':
        model = ''
        clean = rusty = 0
        if conv_bool(cube['newskins']):
            cube_type = conv_int(cube['cubetype'])
            if cube_type != 6:  # Used for custom cubes, no error.
                try:
                    model, clean, rusty = CUBE_MODELS_FOR_TYPE[cube_type]
                except KeyError:
                    LOGGER.warning(
                        'Cube "{}" at ({}) has unknown cube type {}!',
                        cube['targetname'], cube['origin'], cube_type,
                    )
        else:
            # Old skin-based lookup
            cube_skin = conv_int(cube['skin'])
            try:
                model, clean, rusty = CUBE_MODELS_FOR_SKIN[cube_skin]
            except KeyError:
                LOGGER.warning(
                    'Cube "{}" at ({}) has unknown old-style cube skin {}!',
                    cube['targetname'], cube['origin'], cube_skin,
                )
        if model:
            potentials.append((model, rusty if conv_bool(cube['skintype']) else clean))
    elif cube['classname'] == 'prop_monster_box':
        # Hardcoded model. Prefer box form.
        potentials += [
            ('models/npcs/monsters/monster_a_box.mdl', 0),
            ('models/npcs/monsters/monster_a.mdl', 0),
        ]

    for model, skin in potentials:
        try:
            return vac_objects[group, model, skin]
        except KeyError:
            pass
    raise LookupError('No matching object!')
