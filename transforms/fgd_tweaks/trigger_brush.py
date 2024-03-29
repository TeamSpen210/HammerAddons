"""Small tweaks to different entities to make them easier to use.


"""
from srctools import conv_int

from hammeraddons.bsp_transform import trans, Context


@trans('FGD - trigger_brush')
def trigger_brush_input_filters(ctx: Context) -> None:
    """Copy spawnflags on top of the keyvalue.

    This way you get checkboxes you can easily control.
    """
    for ent in ctx.vmf.by_class['trigger_brush']:
        if conv_int(ent['spawnflags']):
            ent['InputFilter'] = ent['spawnflags']
