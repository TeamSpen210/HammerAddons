"""Implements 'forgiving laserfields', which makes them knock back players instead of instantly killing them.

This requires identifying the thinnest brush side, then creating trigger_catapult duplicates in both
directions.
"""
from operator import itemgetter

import itertools

from hammeraddons.bsp_transform import trans, Context, ent_description
from srctools import conv_float, Entity, conv_int, logger, Vec, Output, Matrix
from srctools.bsp import BModel


LOGGER = logger.get_logger(__name__)


@trans('P2 Forgiving Laserfields', priority=10)  # After FGD damage types edits.
def comp_trigger_coop(ctx: Context) -> None:
    """Creates forgiving laserfields."""
    # To create the appropriate visuals, we hurt the player with 1 HP of damage.
    # This caches the names of point_hurts we spawn, depending on the damage bits.
    hurter_cache: dict[int, str] = {}

    for hurt in ctx.vmf.by_class['trigger_hurt']:
        dist = conv_float(hurt.pop('ha_knockback_dist'))
        if dist > 0:
            add_triggers(ctx, hurter_cache, hurt, dist)


def add_triggers(
    ctx: Context,
    hurter_cache: dict[int, str],
    hurt: Entity, shift_dist: float,
) -> None:
    """Modify a trigger."""
    desc = ent_description(hurt)

    # First, we need to determine the axis to spawn along. We'll look for a pair of
    # faces in a brush facing opposite directions, which are the closest together. If not paired
    # we'll just ignore, which allows things like bevelled corners. If multiple axes are close
    # or none found, error.
    try:
        bmodel: BModel = ctx.bsp.bmodels[hurt]
    except KeyError as exc:
        # Could handle this by checking mins/maxes, but this requires map manipulation to create,
        # Hammer doesn't let you make a point ent from something defined as a brush ent. So don't
        # bother.
        raise ValueError(
            f'Forgiving laserfields enabled for {desc}, '
            f'but this is a point entity!',
        ) from exc

    brush_axes: list[tuple[float, Vec]] = []
    # If in an instance, it might have angles set, which will change the orientation after spawning.
    # We want to apply that to our offsets.
    orient = Matrix.from_angstr(hurt['angles'])

    # We'll just compare every plane to all others. Should be fine, we expect the hurt to just be
    # a few prisms. Planes are stored in their own lump, so we can deduplicate via object identity.
    planes = {face.plane for face in bmodel.faces}
    for plane1, plane2 in itertools.product(planes, planes):
        if id(plane1) >= id(plane2) or Vec.dot(plane1.normal, plane2.normal) > -0.99:
            # Compare IDs so we only check each pair once, and also skip self-compares at the same time.
            # Also skip non-opposite faces.
            continue
        # Distances are inverted, so addition = difference.
        brush_axes.append((plane1.dist + plane2.dist, plane1.normal @ orient))
    brush_axes.sort(key=itemgetter(0))
    if not brush_axes:
        raise ValueError(
            f'Forgiving laserfields enabled for {desc}, '
            f'but no opposing faces were detected!'
        )
    # If multiple brushes are present (for non-convex field shapes, or just redundant brushes),
    # we might have copies of axes. Allow duplicates if they're aligned.
    (best_dist, best_norm) = brush_axes[0]
    for (dist, norm) in brush_axes[1:]:
        if abs(dist - best_dist) > 1:
            break  # Hit a larger value, all the rest are fine.
        if abs(Vec.dot(best_norm, norm)) < 0.99:
            candiates = '\n'.join(
                f'Dir={norm}, dist={dist}'
                for dist, norm in brush_axes
            )
            raise ValueError(
                f'Forgiving laserfields enabled for {desc}, '
                f'but multiple candiate directions were detected:\n{candiates}'
            )
    offsets = [shift_dist * best_norm, -shift_dist * best_norm]
    if best_norm.x < -0.1 or best_norm.y < -0.1 or best_norm.z < -0.1:
        # Make order consistent across compiles.
        offsets.reverse()

    LOGGER.debug('Hurt trigger {} using offsets {}, {}', desc, *offsets)

    # Create the point_hurt to use.
    damage_type = conv_int(hurt['damagetype'])
    # Strip various damage flags.
    damage_type &= ~(
        4096 |  # Never Gib
        8192 |  # Always Gib
        4194304 |  # Remove (No Ragdoll)
        134217728  # Blast Surface
    )
    # Our hurters shouldn't apply extra force.
    damage_type |= 2048
    try:
        global_hurter = hurter_cache[damage_type]
    except KeyError:
        global_hurter_ent = ctx.vmf.create_ent(
            classname='point_hurt',
            targetname='_forgiving_hurt',
            Damage=1,  # Just for visuals and sounds.
            DamageType=damage_type,
            DamageTarget='!activator',  # Hurt the triggering player.
            DamageRadius=1,  # Target makes this unused.
            origin='0 0 0',
        ).make_unique()
        global_hurter = hurter_cache[damage_type] = global_hurter_ent['targetname']

    origin = Vec.from_str(hurt['origin'])
    for offset in offsets:
        catapult = ctx.vmf.create_ent(
            'trigger_catapult',
            spawnflags=1,  # Players only.
            origin=origin + offset,
            physicsSpeed=0,
            playerSpeed=125,
            launchDirection=offset.to_angle(),
        )
        catapult.add_out(Output('OnCatapulted', global_hurter, 'Hurt'))
        # Reuse the original brush model.
        ctx.bsp.bmodels[catapult] = bmodel
        # Transfer only some keys. If VScript etc is set, not really safe to copy.
        for key in [
            'targetname', 'parentname', 'spawnflags', 'angles', 'startdisabled', 'filtername',
        ]:
            catapult[key] = hurt[key]
