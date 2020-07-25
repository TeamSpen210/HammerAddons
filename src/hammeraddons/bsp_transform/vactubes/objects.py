from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Set
import os.path

import srctools
from srctools.bsp_transform.packing import make_precache_prop
from srctools.packlist import PackList

from srctools import Vec, VMF


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
        return '<Vac Object "{}">'.format(os.path.basename(self.model_vac))


def parse(vmf: VMF, pack: PackList) -> Tuple[
    int,
    Dict[Tuple[str, str, int], VacObject],
    Dict[str, str],
]:
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
        ent.keys = {
            'model': ent['model']
        }
        make_precache_prop(ent)

        if obj.model_drop:
            cube_objects[
                obj.group,
                obj.model_drop.replace('\\', '/'),
                obj.skin_drop,
            ] = obj

    # Generate and pack the vactube object scripts.
    # Each group is the same, so it can be shared among them all.
    codes = {}
    for group in sorted(vac_objects):
        code = []
        for i, obj in enumerate(vac_objects[group]):
            if obj.model_drop:
                model_code = f'"{obj.model_drop}"'
            else:
                model_code = 'null'
            code.append(
                f'{obj.id} <- obj("{obj.model_vac}", {obj.skin_vac}, '
                f'{model_code}, {obj.weight}, "{obj.offset}", {obj.skin_tv});'
            )
        codes[group] = pack.inject_vscript('\n'.join(code))

    return len(vac_objects), cube_objects, codes
