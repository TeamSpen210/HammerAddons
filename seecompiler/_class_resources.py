"""For each entity class, specify hardcoded resources.

Those are ones that don't simply appear in fgd.
"""
from typing import Iterator, Callable, Tuple, Union, List, Dict

from seecompiler.packlist import FileType
from srctools import Entity, conv_int

__all__ = ['CLASS_RESOURCES']

#  For various entity classes, we know they require hardcoded files.
# List them here - classname -> [(file, type), ...]

_cls_res_type = Dict[str, Union[
    List[Union[str, Tuple[str, FileType]]],
    Callable[[Entity], Iterator[str]],
]]

CLASS_RESOURCES = {}  # type: _cls_res_type


def res(cls, *items):
    """Add a resource to class_resources."""
    CLASS_RESOURCES[cls] = list(items)


def cls_func(func):
    """Save a function to do special checks for a classname."""
    CLASS_RESOURCES[func.__name__] = func
    return func


# *** Base entities  / HL2 ***
res('env_screeneffect',
    'materials/effects/stun.vmt',
    'materials/effects/introblur.vmt',
    )

res('env_fire_trail',
    'materials/sprites/flamelet1.vmt',
    'materials/sprites/flamelet2.vmt',
    'materials/sprites/flamelet3.vmt',
    'materials/sprites/flamelet4.vmt',
    'materials/sprites/flamelet5.vmt',
    'materials/particle/particle_smokegrenade.vmt',
    'materials/particle/particle_noisesphere.vmt',
    )

res('env_steam',
    'materials/particle/particle_smokegrenade.vmt',
    'materials/sprites/heatwave.vmt',
)
CLASS_RESOURCES['env_steamjet'] = CLASS_RESOURCES['env_steam']

res('env_starfield',
    'materials/effects/spark_noz.vmt',
    )

res('func_dust',
    'materials/particle/sparkles.vmt',
    )

res('func_tankchange',
    ('FuncTrackChange.Blocking', FileType.GAME_SOUND),
    )

res('npc_combine_cannon',
    'models/combine_soldier.mdl',
    'materials/effects/bluelaser1.vmt',
    'materials/sprites/light_glow03.vmt',
    ('NPC_Combine_Cannon.FireBullet', FileType.GAME_SOUND),
)

@cls_func
def func_breakable_surf(ent: Entity):
    yield 'models/brokenglass_piece.mdl'

    surf_type = conv_int(ent['surfacetype'])
    if surf_type == 1:  # Tile
        yield from (
            'models/brokentile/tilebroken_03a.mdl',
            'models/brokentile/tilebroken_03b.mdl',
            'models/brokentile/tilebroken_03c.mdl',
            'models/brokentile/tilebroken_03d.mdl',

            'models/brokentile/tilebroken_02a.mdl',
            'models/brokentile/tilebroken_02b.mdl',
            'models/brokentile/tilebroken_02c.mdl',
            'models/brokentile/tilebroken_02d.mdl',

            'models/brokentile/tilebroken_01a.mdl',
            'models/brokentile/tilebroken_01b.mdl',
            'models/brokentile/tilebroken_01c.mdl',
            'models/brokentile/tilebroken_01d.mdl',
        )
    elif surf_type == 0:  # Glass
        yield from (
            'models/brokenglass/glassbroken_solid.mdl',
            'models/brokenglass/glassbroken_01a.mdl',
            'models/brokenglass/glassbroken_01b.mdl',
            'models/brokenglass/glassbroken_01c.mdl',
            'models/brokenglass/glassbroken_01d.mdl',
            'models/brokenglass/glassbroken_02a.mdl',
            'models/brokenglass/glassbroken_02b.mdl',
            'models/brokenglass/glassbroken_02c.mdl',
            'models/brokenglass/glassbroken_02d.mdl',
            'models/brokenglass/glassbroken_03a.mdl',
            'models/brokenglass/glassbroken_03b.mdl',
            'models/brokenglass/glassbroken_03c.mdl',
            'models/brokenglass/glassbroken_03d.mdl',
        )

@cls_func
def move_rope(ent: Entity):
    old_shader_type = conv_int(ent['RopeShader'])
    if old_shader_type == 0:
        yield 'materials/cable/cable.vmt'
    elif old_shader_type == 1:
        yield 'materials/cable / rope.vmt'
    else:
        yield 'materials/cable/chain.vmt'
    yield 'materials/cable/rope_shadowdepth.vmt'


res('npc_vehicledriver',
    'models/roller_vehicledriver.mdl',
    )

res('point_spotlight',
    'materials/sprites/light_glow03.vmt',
    'materials/sprites/glow_test02.vmt',
    )

res('vgui_screen',
    'materials/engine/writez.vmt',
    )

# *** Portal 1/2 ***

res('point_energy_ball_launcher',
    'models/effects/combineball.mdl',
    'materials/effects/eball_finite_life.vmt',
    'materials/effects/eball_infinite_life.vmt',

    'sound/weapons/physcannon/energy_bounce1.wav',
    'sound/weapons/physcannon/energy_bounce2.wav',
    'sound/weapons/physcannon/energy_disintegrate4.wav',
    'sound/weapons/physcannon/energy_disintegrate5.wav',
    'sound/weapons/physcannon/energy_sing_explosion2.wav',
    'sound/weapons/physcannon/energy_sing_flyby1.wav',
    'sound/weapons/physcannon/energy_sing_flyby2.wav',
    'sound/weapons/physcannon/energy_sing_loop4.wav',
    )

res('prop_button',
    'models/props/switch001.mdl'
    )

CLASS_RESOURCES['prop_energy_ball'] = CLASS_RESOURCES['point_energy_ball_launcher']
