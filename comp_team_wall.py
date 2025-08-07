"""Alias for func_team_wall."""
from hammeraddons.bsp_transform import trans, Context


@trans('comp_team_wall')
def trigger_brush_input_filters(ctx: Context) -> None:
    """Alias for func_team_wall. 

    This allows func_team_wall to be used as either a point entity or a brush entity, as it is
    designed to be usable as either, but FGD syntax prohibits this if they share a name.
    """
    for ent in ctx.vmf.by_class['comp_team_wall']:
        ent['classname'] = "func_team_wall"