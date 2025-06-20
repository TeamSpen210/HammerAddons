"""Portal-2 specific transformations."""
from srctools import Output, conv_bool, conv_int, VMF
import srctools.logger

from hammeraddons.bsp_transform import trans, Context, ent_description

LOGGER = srctools.logger.get_logger(__name__)


@trans('Fix Laser Catcher Skins')
def laser_catcher_skins(ctx: Context) -> None:
    """Fix Valve's bug where reloading saves causes lasers to get their skin wrong."""
    for ent in ctx.vmf.by_class['prop_laser_catcher']:
        if not conv_bool(ent.pop('src_fix_skins'), True):
            continue

        deact_skin, act_skin = ('2', '3') if ent['SkinType'] == '1' else ('0', '1')

        # Look for outputs which do this already.
        name = ent['targetname']

        has_act = has_deact = False
        for out in ent.outputs:
            if has_act and has_deact:
                break
            if out.target == name or out.target == '!self':
                if out.input.casefold() == 'skin':
                    if out.params == act_skin:
                        has_act = True
                    elif out.params == act_skin:
                        has_act = True

        if not has_act:
            ent.add_out(Output('OnPowered', '!self', 'Skin', act_skin))
        if not has_deact:
            ent.add_out(Output('OnUnPowered', '!self', 'Skin', deact_skin))


def needs_paint(vmf: VMF) -> bool:
    """Check if we have paint."""
    for ent_cls in [
        'prop_paint_bomb',
        'paint_sphere',
        'weapon_paintgun',  # Not in retail but someone might add it.
    ]:
        if vmf.by_class[ent_cls]:
            return True

    for ent in vmf.by_class['info_paint_sprayer']:
        # Special case, this makes sprayers only render visually, which
        # works even without the value set.
        if not conv_bool(ent['DrawOnly']):
            return True

    for ent_cls in [
        'prop_weighted_cube',
        'prop_physics_paintable',
    ]:
        for ent in vmf.by_class[ent_cls]:
            # If the cube is bouncy, enable paint.
            if conv_int(ent['paintpower', '4'], 4) != 4:
                return True
    return False


@trans('Force Paint in Map')
def force_paintinmap(ctx: Context) -> None:
    """If paint entities are present, set paint in map to true."""
    # Already set, don't bother confirming.
    if conv_bool(ctx.vmf.spawn['paintinmap']):
        return

    if needs_paint(ctx.vmf):
        ctx.vmf.spawn['paintinmap'] = '1'
        # Ensure we have some blobs.
        if conv_int(ctx.vmf.spawn['maxblobcount']) == 0:
            ctx.vmf.spawn['maxblobcount'] = '250'


@trans('Precache P2 Light Bridge')
def precache_light_bridge(ctx: Context) -> None:
    """Ensure light bridges have the particle precached."""

    for bridge in ctx.vmf.by_class['prop_wall_projector']:
        if conv_bool(bridge['StartEnabled', '0']):
            return  # Starts on, no need.
        break
    else:
        # No bridges in the map.
        return

    for part in ctx.vmf.by_class['info_particle_system']:
        # Check for users already fixing the problem.
        if part['effect_name'].casefold() == 'projected_wall_impact':
            return

    ctx.vmf.create_ent(
        classname='info_particle_system',
        origin='-15872 -15872 -15872',
        effect_name='projected_wall_impact',
        start_active='0',
    )


@trans('Fix Button Collision')
def fix_button_collision(ctx: Context) -> None:
    """Portal button entities have strange collision behaviour.

    Unlike all other props solid is a bool, so the normal solid=6(vphysics) makes them nonsolid.
    This fixes that mistake in maps, since 0 is the correct way to make it nonsolid.
    """
    for cls in [
        'prop_contraption_cube_button',
        'prop_floor_ball_button',
        'prop_floor_button',
        'prop_floor_cube_button',
        'prop_under_floor_button',
    ]:
        for btn in ctx.vmf.by_class[cls]:
            if conv_int(btn['solid']) == 6:
                LOGGER.warning(
                    '{} has solid=6, this will be non-solid. Fixing.',
                    ent_description(btn),
                )
                # By default they're solid, just remove.
                del btn['solid']
