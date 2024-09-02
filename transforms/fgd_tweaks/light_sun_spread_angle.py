"""This keyvalue is case-sensitive, force the required casing."""
from hammeraddons.bsp_transform import Context, trans


@trans('FGD - Fix Sun Spread Angle')
def light_sun_spread_angle(ctx: Context) -> None:
    """Force case-sensitivity on this keyvalue."""
    for ent in ctx.vmf.by_class['light_environment'] | ctx.vmf.by_class['light_directional']:
        ent['SunSpreadAngle'] = ent.pop('sunspreadangle')
        print(dict(ent.items()))
