"""Tweak material_modify_control to avoid the variable showing up in instances."""
from hammeraddons.bsp_transform import trans, Context


@trans('FGD - material_modify_control')
def material_modify_control(ctx: Context) -> None:
    """Prepend $ to mat-mod-control variable keyvalues if required.

    This allows Hammer to not detect this as a fixup variable.
    """
    for ent in ctx.vmf.by_class['material_modify_control']:
        var_name = ent['materialvar']
        if var_name and not var_name.startswith(('$', '%')):
            ent['materialvar'] = '$' + var_name
