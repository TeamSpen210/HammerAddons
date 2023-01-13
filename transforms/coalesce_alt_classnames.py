"""Replace deprecated classnames with the correct versions."""
from hammeraddons.bsp_transform import Context, trans
from srctools.fgd import EntityDef


@trans('Fix alternate classnames')
def fix_alt_classnames(ctx: Context) -> None:
    """A bunch of entities has additional alternate names.

    Fix that by coalescing them all to one name.
    """
    for clsname, entset in list(ctx.vmf.by_class.items()):
        try:
            ent_def = EntityDef.engine_def(clsname)
        except KeyError:
            continue
        if ent_def.is_alias:
            try:
                [base] = ent_def.bases
            except ValueError:
                continue
            for ent in entset:
                ent['classname'] = base.classname if isinstance(base, EntityDef) else base
