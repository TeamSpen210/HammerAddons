"""A custom logic entity to correctly sequence portal piston platforms."""
import itertools
import io

from srctools import Matrix, Output, Vec, conv_bool, conv_float, conv_int, logger, Entity
import attrs

from hammeraddons.bsp_transform import Context, ent_description, trans

LOGGER = logger.get_logger(__name__)


@attrs.define
class Piston:
    """Info about each piston segment."""
    ent: Entity
    pos: Vec  # origin = spawn position.
    start: Vec  # SetPosition(0) or Close
    end: Vec  # SetPosition(1) or Open
    fraction: float
    inverted: bool = False  # If true, this retracts in the end position.
    extended: bool = False  # Whether it starts extended.

    @classmethod
    def calculate(cls, ctx: Context, ent: Entity) -> 'Piston':
        """Calculate the two endpoints of this brush ent."""
        origin = Vec.from_str(ent['origin'])
        move_dir = Matrix.from_angstr(ent['movedir']).forward()
        orient = Matrix.from_angstr(ent['angles'])

        match ent['classname'].casefold():
            case 'func_movelinear':
                move_dist = conv_float(ent['movedistance'])
            case 'func_door':
                move_dist = 0.0
            case unknown:
                raise ValueError(f'Unknown segment classname {unknown}?')
        if move_dist <= 0.0:
            # func_movelinear only uses bbox if move distance is blank, doors always use it.
            bmodel = ctx.bsp.bmodels[ent]
            size = bmodel.maxes - bmodel.mins
            move_dist = Vec.dot(move_dir, size) - conv_float(ent['lip'], 0.0)
        fraction = conv_float(ent['startposition'])
        move_dir @= orient
        start_pos = origin - move_dir * fraction * move_dist
        end_pos = start_pos + move_dir * move_dist
        return cls(ent, origin, start_pos, end_pos, fraction)


@trans('Portal Piston Platforms')
def piston_platform(ctx: Context) -> None:
    """A custom logic entity to correctly sequence portal piston platforms."""
    for ent in ctx.vmf.by_class['comp_piston_platform']:
        generate_platform(ctx, ent)


def generate_platform(ctx: Context, logic_ent: Entity) -> None:
    """Generate a piston platform."""
    desc = ent_description(logic_ent)

    # Locate and validate the piston segments.
    pistons = {}
    for key, ent_name in logic_ent.items():
        if not key.casefold().startswith('piston'):
            continue
        if '#' in key:  # TODO: Handle this in srctools.
            key = key.split('#', 1)[0]
        try:
            ind = int(key.removeprefix('piston'))
        except (TypeError, ValueError):
            continue  # Not a piston key?

        for ent in ctx.vmf.search(ent_name):
            if ent['classname'].casefold() not in ('func_movelinear', 'func_door'):
                # Warn and ignore, could be another ent with the same name or something.
                LOGGER.warning(
                    '{} has unknown segment #{} {}. '
                    'Expected func_movelinear or func_door.',
                    desc, ind, ent_description(ent),
                )
                continue
            if ind in pistons:
                # Can't continue, ambiguous as to the order.
                raise ValueError(
                    f'{desc} located multiple entities '
                    f'for piston segment "{key}" = "{ent_name}"!'
                )
            pistons[ind] = Piston.calculate(ctx, ent)
        if ind not in pistons:
            # Just a warning, so you can start with a prefab, remove unnecessary bits and it'll
            # still work.
            LOGGER.warning(
                '{} could not find piston segment "{}"!',
                desc, key,
            )

    if not pistons:
        raise ValueError(f'{desc} has no piston segments!')
    # Check indices are contiguous.
    lowest = min(pistons)
    highest = max(pistons)
    indices = range(lowest, highest + 1)
    if missing := pistons.keys() - set(indices):
        raise ValueError(
            f'{desc} has missing piston segments. '
            f'Segments span {indices.start} - {indices.stop-1}, but {sorted(missing)} are missing.'
        )
    LOGGER.debug('Piston {} spans range {} - {}', desc, indices.start, indices.stop-1)

    # Calculate which are extended/retracted.
    # Find the combo of top/bottom segments that have the biggest difference, which gives the fully
    # extended position. Use that to determine extension direction, then determine inversion from
    # there, plus whether it starts closest to start or end position.
    if len(pistons) == 1:
        # Nothing to determine, assume it's not inverted.
        pist = pistons[indices[0]]
        pist.inverted = False
        pist.extended = pist.fraction > 0.5
        extend_dir = (pist.end - pist.start).norm()
    else:
        base, tip = max(
            itertools.product(
                [pistons[lowest].start, pistons[lowest].end],
                [pistons[highest].start, pistons[highest].end],
            ),
            key=lambda t: (t[1] - t[0]).mag_sqr()
        )
        extend_dir = (tip - base).norm()
        for pist in pistons.values():
            pist.inverted = Vec.dot(pist.end - pist.start, extend_dir) < 0
            pist.extended = (pist.fraction > 0.5) ^ pist.inverted

    use_vscript = conv_bool(logic_ent['usevscript'])

    underside_fizz: Entity | None = None
    underside_hurt: Entity | None = None
    # Remove duplicates, then remove blanks/unset.
    for ent_name in filter(None, {
        logic_ent['underside_fizz'].casefold(),
        logic_ent['underside_hurt'].casefold(),
    }):
        for ent in ctx.vmf.search(ent_name):
            cls = ent['classname'].casefold()
            # Use a multiple to fire Break/self-destruct inputs,
            # or just a fizzler to fizzle everything.
            if cls in ('trigger_multiple', 'trigger_portal_cleanser'):
                if underside_fizz is not None:
                    raise ValueError(
                        '{} found duplicate underside hurt triggers {} and {}!',
                        desc,
                        ent_description(underside_hurt), ent_description(ent)
                    )
                underside_fizz = ent
            elif cls == 'trigger_hurt':
                if underside_hurt is not None:
                    raise ValueError(
                        '{} found duplicate underside hurt triggers {} and {}!',
                        ent_description(logic_ent),
                        ent_description(underside_hurt), ent_description(ent)
                    )
                underside_hurt = ent
            else:
                LOGGER.warning(
                    'Unexpected classname for undersize fizzler/hurt "{}" for piston {}?',
                    ent['classname'], desc
                )

    if conv_bool(logic_ent['underside_auto', '1'], True):
        # Configure/create both entities.
        # We turn on/off both simultaneously so it doesn't matter whether they
        # have the same name or not.
        if underside_fizz is None and underside_hurt is not None:
            underside_fizz = ctx.vmf.create_ent(
                'trigger_multiple',
                origin=underside_hurt['origin'],
                targetname=underside_hurt['targetname'],
            )
            ctx.bsp.bmodels[underside_fizz] = ctx.bsp.bmodels[underside_hurt]
        elif underside_hurt is None and underside_fizz is not None:
            underside_hurt = ctx.vmf.create_ent(
                'trigger_hurt',
                origin=underside_fizz['origin'],
                targetname=underside_fizz['targetname'],
            )
            ctx.bsp.bmodels[underside_hurt] = ctx.bsp.bmodels[underside_fizz]

        # Configure each, if they're present. Could be missing if both are.
        if underside_fizz is not None:
            configure_fizzler(underside_fizz, desc)
        if underside_hurt is not None:
            underside_hurt['spawnflags'] = 1  # Only players make sense.
            if conv_int(underside_hurt['damage']) < 100:
                underside_hurt['damage'] = 100  # Instakill
            if conv_int(underside_hurt['damagetype']) == 0:
                # If generic, make it CRUSH. Otherwise, user picked for a reason?
                underside_hurt['damagetype'] = 1

    if use_vscript:
        gen_logic_vscript(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            underside_fizz=underside_fizz,
            underside_hurt=underside_hurt,
        )
    else:
        gen_logic_branches(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            underside_fizz=underside_fizz,
            underside_hurt=underside_hurt,
        )


def configure_fizzler(fizz: Entity, desc: str) -> None:
    """Configure the fizzler."""
    flags = conv_int(fizz['spawnflags'])
    # Add physics objects, remove players.
    flags |= 8
    flags &= ~1
    fizz['spawnflags'] = flags

    cls = fizz['classname'].casefold()
    if cls == 'trigger_portal_cleanser':
        # It's a fizzler, just turn off visibility/scanlines.
        fizz['visible'] = False
        fizz['usescanline'] = False
        return
    # Otherwise, it's a trigger_multiple and we need to check outputs.
    assert cls == 'trigger_multiple', cls
    # If any OnStartTouch/OnTrigger outputs are present, assume the user has configured
    # them.
    for out in fizz.outputs:
        if out.output.casefold() in ('ontrigger', 'onstarttouch'):
            LOGGER.info('Outputs found on underside fizzler "{}", assuming these fizzle.')
            return
    # Order these inputs by priority, each kills the ent so the rest won't fire.
    fizz.add_out(
        Output('OnStartTouch', '!activator', 'SelfDestructImmediately'),  # Turrets
        Output('OnStartTouch', '!activator', 'Dissolve', 0.05),  # Cubes
        Output('OnStartTouch', '!activator', 'Break', 0.1),  # Any props (gibs)
    )


def gen_logic_vscript(
    *,
    ctx: Context, logic_ent: Entity,
    indices: range, pistons: dict[int, Entity],
    underside_fizz: Entity | None,
    underside_hurt: Entity | None,
) -> None:
    """Generate VScript logic for pistons."""
    # Generate VScript.
    logic_ent['classname'] = 'logic_script'
    code = io.StringIO()
    code.write(f'MAX_IND = {indices.stop - 1};\n')


def gen_logic_branches(
    *,
    ctx: Context, logic_ent: Entity,
    indices: range, pistons: dict[int, Entity],
    underside_fizz: Entity | None,
    underside_hurt: Entity | None,
) -> None:
    """Generate basic logic for pistons, using logic_branch."""
    logic_ent.remove()
