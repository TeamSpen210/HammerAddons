from typing import Optional, Tuple, List, Dict
import os.path

import srctools
from srctools.packlist import PackList

from srctools import Vec, VMF


class VacObject:
    """An object that can appear in vactubes."""
    def __init__(
        self,
        obj_id: str,
        model_vac: str,
        model_drop: Optional[str],
        offset: Vec,
        weight: int=1,
        skin_tv: int=0,
        skin_drop: int=0,
        skin_vac: int=0,
    ) -> None:
        self.id = obj_id
        self.model_vac = model_vac  # Model for in vactubes
        self.model_drop = model_drop  # If a cube, the real cube model.
        self.weight = weight
        self.offset = offset
        self.skin_tv = skin_tv  # If set, switch scanner TVs to this while passing.
        self.skin_vac = skin_vac
        self.skin_drop = skin_drop

    def __repr__(self) -> str:
        return '<Vac Object "{}">'.format(os.path.basename(self.model_vac))


def parse(vmf: VMF, pack: PackList) -> Tuple[Dict[Tuple[str, int], VacObject], str]:
    """Parse out the cube objects from the map."""
    cube_objects: Dict[Tuple[str, int], VacObject] = {}
    vac_objects: List[VacObject] = []

    for i, ent in enumerate(vmf.by_class['comp_vactube_object']):
        offset = Vec.from_str(ent['origin']) - Vec.from_str(ent['offset'])
        obj = VacObject(
            f'obj_{i:x}',
            ent['model'],
            ent['cube_model'],
            offset,
            srctools.conv_int(ent['weight']),
            srctools.conv_int(ent['tv_skin']),
            srctools.conv_int(ent['cube_skin']),
            srctools.conv_int(ent['skin']),
        )
        vac_objects.append(obj)
        # Convert the ent into a precache command
        ent['classname'] = 'comp_precache_model'
        if obj.model_drop:
            cube_objects[
                obj.model_drop.replace('\\', '/'),
                obj.skin_drop,
            ] = obj

    # Generate and pack the vactube object info.
    code = []
    for i, obj in enumerate(vac_objects):
        if obj.model_drop:
            model_code = f'"{obj.model_drop}"'
        else:
            model_code = 'null'
        code.append(
            f'{obj.id} <- obj("{obj.model_vac}", {obj.skin_vac}, '
            f'{model_code}, {obj.weight}, "{obj.offset}", {obj.skin_tv});\n'
        )
    return cube_objects, pack.inject_vscript(''.join(code))