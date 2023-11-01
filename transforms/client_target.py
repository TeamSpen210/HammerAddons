"""Automatically enable the clientside flag on info_target entities."""
from hammeraddons.bsp_transform import Context, trans
from srctools import Entity, conv_int
from srctools.logger import get_logger

LOGGER = get_logger(__name__)

KEYVALUES = [
    ('info_particle_system', [f'cpoint{x}' for x in range(1, 64)]),
    ('env_instructor_hint', ['hint_target']),
]


@trans('Enable info_target clientside')
def clientside_info_target(ctx: Context) -> None:
    """Automatically enable the clientside flag on info_target entities.
    """
    for clsname, keys in KEYVALUES:
        ent: Entity
        for ent in ctx.vmf.by_class[clsname]:
            for key in keys:
                value = ent[key]
                if not value:
                    continue
                for target in ctx.vmf.search(value):
                    if target['classname'].casefold() == 'info_target':
                        spawnflags = conv_int(target['spawnflags'])
                        target['spawnflags'] = spawnflags | 1  # Transmit to client (respect PVS)
