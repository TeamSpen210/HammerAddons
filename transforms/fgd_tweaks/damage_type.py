"""Break up the damage type keyvalue into multiple, to allow fully specifying all the values."""
from srctools import conv_int

from hammeraddons.bsp_transform import trans, Context


@trans('FGD - damage types')
def damage_types(ctx: Context) -> None:
    """Break up the damage type keyvalue into multiple, to allow fully specifying all the values."""
    for ent in ctx.vmf.entities:
        if 'damagetype' not in ent:  # Not relevant
            continue
        damage = conv_int(ent['damagetype'])
        for i in range(1, 11):
            damage |= conv_int(ent[f'damageor{i}'])
        ent['damagetype'] = int(damage)
