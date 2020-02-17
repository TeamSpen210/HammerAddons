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
        tv_skin: int=0,
    ) -> None:
        self.id = obj_id
        self.model_vac = model_vac  # Model for in vactubes
        self.model_drop = model_drop  # If a cube, the real cube model.
        self.weight = weight
        self.offset = offset
        self.tv_skin = tv_skin  # If set, switch scanner TVs to this while passing.

    def __repr__(self) -> str:
        return '<Vac Object "{}">'.format(os.path.basename(self.model_vac))


def parse(vmf: VMF, pack: PackList) -> Tuple[Dict[str, VacObject], str]:
    cube_objects: Dict[str, VacObject] = {}
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
        )
        vac_objects.append(obj)
        # Convert the ent into a precache command
        ent['classname'] = 'comp_precache_model'
        if obj.model_drop:
            cube_objects[obj.model_drop.replace('\\', '/')] = obj

    # Generate and pack the vactube object info.
    code = []
    for i, obj in enumerate(vac_objects):
        if obj.model_drop:
            model_code = f'"{obj.model_drop}"'
        else:
            model_code = 'null'
        code.append(
            f'{obj.id} <- obj("{obj.model_vac}", {model_code}, '
            f'{obj.weight}, "{obj.offset}", '
            f'{obj.tv_skin});\n'
        )
    return cube_objects, pack.inject_vscript(''.join(code))
