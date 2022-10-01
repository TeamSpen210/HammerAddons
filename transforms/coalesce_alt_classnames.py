"""Replace deprecated classnames with the correct versions."""
from hammeraddons.bsp_transform import Context, trans
from srctools.packlist import entclass_canonicalise


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
