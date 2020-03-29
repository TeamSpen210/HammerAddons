"""Small tweaks to different entities to make them easier to use.


"""
from srctools import conv_int

from srctools.bsp_transform import trans, Context
from srctools.packlist import ALT_NAMES


@trans('trigger_brush Input Filters')
def trigger_brush_input_filters(ctx: Context) -> None:
    """Copy spawnflags on top of the keyvalue.

    This way you get checkboxes you can easily control.
    """
    for ent in ctx.vmf.by_class['trigger_brush']:
        if conv_int(ent['spawnflags']):
            ent['InputFilter'] = ent['spawnflags']


@trans('Fix alternate classnames')
def fix_alt_classnames(ctx: Context) -> None:
    """A bunch of entities have additional alternate names.

    Fix that by coalescing them all to one name.
    """
    for alt, replacement in ALT_NAMES.items():
        for ent in ctx.vmf.by_class[alt]:
            ent['classname'] = replacement
