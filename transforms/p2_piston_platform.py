"""A custom logic entity to correctly sequence portal piston platforms."""
from collections.abc import Collection

from typing import NoReturn, Self

import itertools
import io

from srctools import Matrix, Output, Vec, conv_bool, conv_float, conv_int, logger, Entity
import attrs

from hammeraddons.bsp_transform.common import strip_cust_keys, ent_description
from hammeraddons.bsp_transform import Context, trans

LOGGER = logger.get_logger(__name__)


@attrs.define
class Piston:
    """Info about each piston segment."""
    ent: Entity
    index: int
    pos: Vec  # origin = spawn position.
    start: Vec  # SetPosition(0) or Close
    end: Vec  # SetPosition(1) or Open
    fraction: float
    inverted: bool = False  # If true, this retracts in the end position.
    extended: bool = False  # Whether it starts extended.

    @classmethod
    def calculate(cls, ctx: Context, ind: int, ent: Entity) -> Self:
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
        return cls(ent, ind, origin, start_pos, end_pos, fraction)


@trans('Portal Piston Platforms')
def piston_platform(ctx: Context) -> None:
    """A custom logic entity to correctly sequence portal piston platforms."""
    # If we need to generate a filter for motion entities, pass that to the rest of the
    # platforms for reuse.
    motion_filter: Entity | None = None
    for ent in ctx.vmf.by_class['comp_piston_platform']:
        motion_filter = generate_platform(ctx, motion_filter, ent)


def generate_platform(ctx: Context, motion_filter: Entity | None, logic_ent: Entity) -> Entity | None:
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
                # Can't continue, ambiguous as to the order. Also means
                # at runtime they can't be targeted individually.
                raise ValueError(
                    f'{desc} located multiple entities '
                    f'for piston segment "{key}" = "{ent_name}"!'
                )
            pistons[ind] = Piston.calculate(ctx, ind, ent)
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
    extend_override = conv_int(logic_ent['position_override'])
    if extend_override != 0:
        # User overrides.
        if extend_override == 1:
            for pist in pistons:
                pist.inverted = pist.extended = False
        elif extend_override == 2:
            for pist in pistons:
                pist.inverted = pist.extended = True
        else:
            raise ValueError(
                f'{desc}: Position override {logic_ent['position_override']} '
                'is invalid, expected 0, 1 or 2.'
            )
    elif len(pistons) == 1:
        # Nothing to determine, assume it's not inverted.
        pist = pistons[indices[0]]
        pist.inverted = False
        pist.extended = pist.fraction > 0.5
    else:
        base, tip = max(
            itertools.product(
                [pistons[lowest].start, pistons[lowest].end],
                [pistons[highest].start, pistons[highest].end],
            ),
            key=lambda t: (t[1] - t[0]).mag_sq()
        )
        extend_dir = (tip - base).norm()
        for pist in pistons.values():
            pist.inverted = Vec.dot(pist.end - pist.start, extend_dir) < 0
            pist.extended = (pist.fraction > 0.5) ^ pist.inverted

    LOGGER.debug('Piston configuration: {}', pistons)

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

    enable_motion_trig: Entity | None = None
    if ent_name := logic_ent['enable_motion_trig']:
        ents = list(ctx.vmf.search(ent_name))
        if len(ents) > 1:
            raise ValueError(f'{desc} found multiple enalbe-motion triggers!')
        elif len(ents) == 1:
            [enable_motion_trig] = ents

    if conv_bool(logic_ent['autoconfig_triggers', '1'], True):
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
            # We only enable if moving to the bottom, so if extended it should be off.
            # If retracted, it'll be covered so it doesn't matter.
            underside_hurt['startdisabled'] = True
            if conv_int(underside_hurt['damage']) < 100:
                underside_hurt['damage'] = 100  # Instakill
            if conv_int(underside_hurt['damagetype']) == 0:
                # If generic, make it CRUSH. Otherwise, user picked for a reason?
                underside_hurt['damagetype'] = 1

        if enable_motion_trig is not None:
            # This should detect only laser cubes, but any cube is close enough.
            if not enable_motion_trig['filtername']:
                if motion_filter is None:
                    # Make the filter.
                    motion_filter = ctx.vmf.create_ent(
                        'filter_activator_class',
                        negated=0,
                        filterclass='prop_weighted_cube',
                    ).make_unique('filter_weighted_cube')
            enable_motion_trig['filtername'] = motion_filter['targetname']
            enable_motion_trig['spawnflags'] = 8  # Physics
            enable_motion_trig['startdisabled'] = 1  # Only turns on briefly.
            for out in enable_motion_trig.outputs:
                if (
                    out.output.casefold() in ('onstarttouch', 'ontrigger')
                    and out.target.casefold() == '!activator'
                    and out.input.casefold() == 'exitdisabledstate'
                ):
                    break  # Already present.
            else:
                enable_motion_trig.add_out(Output(
                    'OnStartTouch', '!activator', 'ExitDisabledState',
                ))

    # Extract names, deduplicating. Both logic types don't need to edit the triggers themselves.
    underside_ents = [
        ent
        for ent in [underside_fizz, underside_hurt]
        if ent is not None
    ]

    if conv_bool(logic_ent['use_vscript']):
        LOGGER.info('Generating VScript logic for piston {}', desc)
        gen_logic_vscript(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            underside_ents=underside_ents,
            enable_motion_trig=enable_motion_trig,
        )
    else:
        LOGGER.info('Generating logic_branch logic for piston {}', desc)
        gen_logic_branches(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            underside_ents=underside_ents,
            enable_motion_trig=enable_motion_trig,
        )
    return motion_filter


def configure_fizzler(fizz: Entity, desc: str) -> None:
    """Configure the fizzler."""
    flags = conv_int(fizz['spawnflags'])
    # Add physics objects, remove players.
    flags |= 8
    flags &= ~1
    fizz['spawnflags'] = flags
    # We only enable if moving to the bottom, so if extended it should be off.
    # If retracted, it'll be covered so it doesn't matter.
    fizz['startdisabled'] = True

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
            LOGGER.info(
                'Outputs found on underside fizzler "{}" for piston {}, assuming these fizzle.',
                ent_description(fizz), desc,
            )
            return
    # Order these inputs by priority, each kills the ent so the rest won't fire.
    fizz.add_out(
        Output('OnStartTouch', '!activator', 'SelfDestructImmediately'),  # Turrets
        Output('OnStartTouch', '!activator', 'Dissolve', delay=0.05),  # Cubes
        Output('OnStartTouch', '!activator', 'Break', delay=0.1),  # Any props (gibs)
    )


def gen_logic_vscript(
    *,
    ctx: Context, logic_ent: Entity,
    indices: range, pistons: dict[int, Piston],
    underside_ents: list[Entity],
    enable_motion_trig: Entity | None,
) -> None:
    """Generate VScript logic for pistons."""
    # Generate init code which configures the piston. We'll also create a function
    # for all possible inputs, so they don't need to be pre-compiled.
    logic_ent['classname'] = 'logic_script'
    logic_name = logic_ent['targetname']
    code = io.StringIO()
    code.write(f'MAX_IND = {indices.stop - 1}\n')
    # Starting height is the number of extended pistons.
    code.write(f'g_target_pos = {sum(pist.extended for pist in pistons.values())}\n')
    # Retract = move lower than the first segment position.
    retract_pos = indices[0] - 1
    code.write(f'function Retract() {{ moveto({retract_pos}) }}\n')
    for pist in pistons.values():
        code.write(f'g_positions[{pist.index}] <- {'POS_UP' if pist.extended else 'POS_DN'}\n')
        code.write(f'g_inverted[{pist.index}] <- {'true' if pist.inverted else 'false'}\n')
        code.write(f'function MoveTo{pist.index}() {{ moveto({pist.index}) }}\n')
        code.write(f'function up{pist.index}() {{ g_positions[{pist.index}]=POS_UP;_up()}}\n')
        code.write(f'function dn{pist.index}() {{ g_positions[{pist.index}]=POS_DN;_dn()}}\n')

        ctx.add_io_remap(logic_name, Output(
            f'MoveTo{pist.index}', logic_name,
            'CallScriptFunction', f'MoveTo{pist.index}',
        ))
        pist.ent.add_out(Output(
            'OnFullyClosed' if pist.inverted else 'OnFullyOpen',
            logic_name, 'CallScriptFunction', f'up{pist.index}',
        ), Output(
            'OnFullyOpen' if pist.inverted else 'OnFullyClosed',
            logic_name, 'CallScriptFunction', f'dn{pist.index}',
        ))

    ctx.add_io_remap(logic_name, Output(
            'Extend', logic_name, 'CallScriptFunction', f'MoveTo{indices[-1]}'
        ), Output('Retract', logic_name, 'CallScriptFunction', 'Retract'),
    )

    code.write('function OnPostSpawn() {\n')
    for pist in pistons.values():
        code.write(f'\tg_pistons[{pist.index}] <- Entities.FindByName(null, "{pist.ent['targetname']}")\n')
    if enable_motion_trig is not None:
        code.write(f'\tenable_motion_trig = Entities.FindByName(null, "{enable_motion_trig['targetname']}")\n')
    if underside_ents:
        if conv_bool(logic_ent['underside_lenient']):
            logic_ent['thinkfunction'] = 'Think'
            code.write('dn_fizz_eager = false\n')
        else:
            code.write('dn_fizz_eager = true\n')

        underside_names = {ent['targetname'] for ent in underside_ents}
        if len(underside_ents) == 2 and len(underside_names) == 1:
            [ent_name] = underside_names
            # Both have the same name, so we need to loop twice.
            code.write(
                f'local first = Entities.FindByName(null, "{ent_name}")\n'
                'dn_fizz_ents.push(first);\n'
                f'dn_fizz_ents.push(Entities.FindByName(first, "{ent_name}"))\n'
            )
        else:
            for ent_name in underside_names:
                code.write(f'dn_fizz_ents.push(Entities.FindByName(null, "{ent_name}"))\n')
        # Leinient fizzlers must always start disabled, VScript turns them on briefly only.
        for ent in underside_ents:
            ent['startdisabled'] = True

    code.write('}\n')

    strip_cust_keys(logic_ent)
    logic_ent['vscripts'] = 'srctools/piston_platform.nut'
    ctx.add_code(logic_ent, code.getvalue())


def gen_logic_branches(
    *,
    ctx: Context, logic_ent: Entity,
    indices: range, pistons: dict[int, Piston],
    underside_ents: list[Entity],
    enable_motion_trig: Entity | None,
) -> None:
    """Generate basic logic for pistons, using logic_branch."""
    logic_ent.remove()

    # For this logic, disallow pistons to start in an inconsistent state.
    if {piston.extended for piston in pistons.values()} == {False, True}:
        raise ValueError(
            f'Piston platform {ent_description(logic_ent)} has some pistons starting extended '
            'and others retracted. VScript is required to support this setup.'
        )

    up_branches = []
    dn_branches = []

    for pist1, pist2 in itertools.pairwise(pistons[i] for i in indices):
        up_branches.append(make_branch(ctx, pist1, pist2, 'up'))
        dn_branches.append(make_branch(ctx, pist2, pist1, 'dn'))
    pist_first = pistons[indices[0]]
    pist_last = pistons[indices[-1]]
    logic_name = logic_ent['targetname']
    ctx.add_io_remap(
        logic_name,
        Output('Extend', pist_first.ent, 'Close' if pist_first.inverted else 'Open'),
        *[Output('Extend', branch, 'SetValue', '1') for branch in up_branches],
        *[Output('Extend', branch, 'SetValue', '0') for branch in dn_branches],
        Output('Retract', pist_last.ent, 'Open' if pist_last.inverted else 'Close'),
        *[Output('Retract', branch, 'SetValue', '0') for branch in up_branches],
        *[Output('Retract', branch, 'SetValue', '1') for branch in dn_branches],
        # MoveTo is allowed to reach the last position only. It's just a synonym.
        Output(f'MoveTo{indices[-1]}', logic_name, 'Extend'),
    )
    if enable_motion_trig is not None:
        ctx.add_io_remap(
            logic_name,
            Output('Extend', enable_motion_trig, 'Enable'),
            Output('Extend', enable_motion_trig, 'Disable', delay=0.1),
            Output('Retract', enable_motion_trig, 'Enable'),
            Output('Retract', enable_motion_trig, 'Disable', delay=0.1),
        )
    # Deduplicate names.
    for name in {ent['targetname'] for ent in underside_ents}:
        ctx.add_io_remap(
            logic_name,
            Output('Extend', name, 'Disable'),
            Output('Retract', name, 'Enable'),
        )

    if underside_ents and conv_bool(logic_ent['underside_lenient']):
        LOGGER.warning(
            'Piston platform {} is requesting lenient underside fizzlers/hurts, '
            'but VScript must be enabled for this feature.',
            ent_description(logic_ent)
        )

    def extension_error(source: Entity, out: Output) -> NoReturn:
        """Error if inputs are used that require VScript."""
        raise ValueError(
            f'Output "{out.input}" cannot be used on piston platform {ent_description(logic_ent)}! '
            f'VScript is required to stop midway. Output originates from {ent_description(source)}.'
        )
    for pos in indices[:-1]:
        ctx.add_io_remap_func(logic_name, f'MoveTo{pos}', extension_error)


def make_branch(ctx: Context, source: Piston, target: Piston, name: str) -> Entity:
    """Create the logic_branch for a direction along with the connections."""
    invert = target.index < source.index
    branch = ctx.vmf.create_ent(
        'logic_branch',
        origin=source.ent['origin'],
    ).make_unique(f'{source.ent["targetname"]}_{name}_branch')
    branch.add_out(
        Output('OnTrue', target.ent, 'Close' if target.inverted ^ invert else 'Open'),
        # Fired if we get here when we're supposed to stop, reverse both.
        Output('OnFalse', target.ent, 'Open' if target.inverted ^ invert else 'Close'),
        Output('OnFalse', source.ent, 'Open' if source.inverted ^ invert else 'Close'),
    )
    source.ent.add_out(Output(
        'OnFullyClosed' if source.inverted ^ invert else 'OnFullyOpen',
        branch, 'Test'
    ))
    return branch
