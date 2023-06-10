"""Optimise brushes used for areaportal windows."""
from hammeraddons.bsp_transform import Context, trans
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


@trans('Optimise Areaportals')
def optimise_areaportals(ctx: Context) -> None:
    """Optimise brushes used for areaportal windows.

    This forces off solidity for func_brush, and clears the physics data.
    """
    for ap_ent in ctx.vmf.by_class['func_areaportalwindow']:
        if ap_ent['target']:
            for fade_ent in ctx.vmf.search(ap_ent['target']):
                if fade_ent['classname'] in ['prop_dynamic', 'prop_dynamic_override']:
                    fade_ent['solid'] = '0'
                elif fade_ent['classname'] in ['func_brush']:
                    fade_ent['solidity'] = '1'  # Never Solid

                try:
                    bmodel = ctx.bsp.bmodels[fade_ent]
                except KeyError:
                    continue  # Model ent?
                else:
                    # For brush models, delete the VPhysics data.
                    bmodel.clear_physics()
