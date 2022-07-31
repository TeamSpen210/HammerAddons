"""Small tweaks to different entities to make them easier to use.


"""
from srctools import conv_int, logger
from srctools.packlist import entclass_canonicalise

from hammeraddons.bsp_transform import trans, Context


LOGGER = logger.get_logger(__name__)


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
    """A bunch of entities has additional alternate names.

    Fix that by coalescing them all to one name.
    """
    for clsname, entset in list(ctx.vmf.by_class.items()):
        canonical = entclass_canonicalise(clsname)
        if canonical != clsname.casefold():
            for ent in entset:
                ent['classname'] = canonical
