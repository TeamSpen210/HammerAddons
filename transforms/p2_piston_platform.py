"""A custom logic entity to correctly sequence portal piston platforms."""
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
            for pist in pistons.values():
                pist.inverted = pist.extended = False
        elif extend_override == 2:
            for pist in pistons.values():
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
        # Pistons are parented to the base, so it doesn't matter where that is.
        # We pick the furthest away position, which is going to be fully extended.
        extend_dir = max([
            pistons[highest].start - pistons[lowest].start,
            pistons[highest].end - pistons[lowest].start,
        ], key=Vec.mag_sq).norm()
        for pist in pistons.values():
            pist.inverted = Vec.dot(pist.end - pist.start, extend_dir) < 0
            pist.extended = (pist.fraction > 0.5) ^ pist.inverted

    LOGGER.debug('Piston configuration: {}', pistons)

    underside_fizz, underside_hurt = process_fizzler_set(ctx, logic_ent, desc, 'underside')
    topside_fizz, topside_hurt = process_fizzler_set(ctx, logic_ent, desc, 'topside')

    enable_motion_trig: Entity | None = None
    if ent_name := logic_ent['enable_motion_trig']:
        ents = list(ctx.vmf.search(ent_name))
        if len(ents) > 1:
            raise ValueError(f'{desc} found multiple enalbe-motion triggers!')
        elif len(ents) == 1:
            [enable_motion_trig] = ents

    if conv_bool(logic_ent['autoconfig_triggers', '1'], True):
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

    if conv_bool(logic_ent['use_vscript']):
        LOGGER.info('Generating VScript logic for piston {}', desc)
        gen_logic_vscript(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            underside_fizz=underside_fizz,
            underside_hurt=underside_hurt,
            topside_fizz=topside_fizz,
            topside_hurt=topside_hurt,
            enable_motion_trig=enable_motion_trig,
        )
    else:
        if topside_hurt or topside_fizz:
            raise ValueError(
                f'Piston {desc} has topside fizzlers/hurts configured, '
                f'but this requires VScript to be useful.'
            )
        LOGGER.info('Generating logic_branch logic for piston {}', desc)
        gen_logic_branches(
            ctx=ctx, logic_ent=logic_ent,
            indices=indices, pistons=pistons,
            # Don't care which is which.
            underside_ents=[underside_fizz, underside_hurt],
            enable_motion_trig=enable_motion_trig,
        )
    return motion_filter


def process_fizzler_set(
    ctx: Context, logic_ent: Entity, desc: str, kind: str,
) -> tuple[Entity | None, Entity | None]:
    """Locate the entities for a fizzler/hurt set, also configuring if necessary."""
    fizz: Entity | None = None
    hurt: Entity | None = None
    # Remove duplicates, then remove blanks/unset.
    for ent_name in filter(None, {
        logic_ent[f'{kind}_fizz'].casefold(),
        logic_ent[f'{kind}_hurt'].casefold(),
    }):
        for ent in ctx.vmf.search(ent_name):
            cls = ent['classname'].casefold()
            # Use a multiple to fire Break/self-destruct inputs,
            # or just a fizzler to fizzle everything.
            if cls in ('trigger_multiple', 'trigger_portal_cleanser'):
                if fizz is not None:
                    raise ValueError(
                        '{} found duplicate {} fizzler triggers {} and {}!',
                        desc, kind,
                        ent_description(fizz), ent_description(ent)
                    )
                fizz = ent
            elif cls == 'trigger_hurt':
                if hurt is not None:
                    raise ValueError(
                        '{} found duplicate {} hurt triggers {} and {}!',
                        desc, kind,
                        ent_description(hurt), ent_description(ent)
                    )
                hurt = ent
            else:
                LOGGER.warning(
                    'Unexpected classname for {} fizzler/hurt "{}" for piston {}?',
                    kind, ent['classname'], desc
                )

    if conv_bool(logic_ent['autoconfig_triggers', '1'], True):
        # Configure/create both entities.
        # We turn on/off both simultaneously so it doesn't matter whether they
        # have the same name or not.
        if fizz is None and hurt is not None:
            fizz = ctx.vmf.create_ent(
                'trigger_multiple',
                origin=hurt['origin'],
                targetname=hurt['targetname'],
            )
            ctx.bsp.bmodels[fizz] = ctx.bsp.bmodels[hurt]
        elif hurt is None and fizz is not None:
            hurt = ctx.vmf.create_ent(
                'trigger_hurt',
                origin=fizz['origin'],
                targetname=fizz['targetname'],
            )
            ctx.bsp.bmodels[hurt] = ctx.bsp.bmodels[fizz]

        # Configure each, if they're present. Could be missing if both are.
        if fizz is not None:
            configure_fizzler(fizz, desc)
        if hurt is not None:
            hurt['spawnflags'] = 1  # Only players make sense.
            # We only enable if moving to the bottom, so if extended it should be off.
            # If retracted, it'll be covered so it doesn't matter.
            hurt['startdisabled'] = True
            if conv_int(hurt['damage']) < 100:
                hurt['damage'] = 100  # Instakill
            if conv_int(hurt['damagetype']) == 0:
                # If generic, make it CRUSH. Otherwise, user picked for a reason?
                hurt['damagetype'] = 1
    return fizz, hurt


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
    underside_fizz: Entity | None,
    underside_hurt: Entity | None,
    topside_fizz: Entity | None,
    topside_hurt: Entity | None,
    enable_motion_trig: Entity | None,
) -> None:
    """Generate VScript logic for pistons."""
    # Generate init code which configures the piston. We'll also create a function
    # for all possible inputs, so they don't need to be pre-compiled.
    logic_name = logic_ent['targetname']
    code = io.StringIO()
    code.write(f'MAX_IND = {indices.stop - 1}\n')
    # Starting height is the number of extended pistons.
    code.write(f'g_target_pos = {sum(pist.extended for pist in pistons.values())}\n')

    snd = logic_ent['snd_start']
    if snd and snd.casefold() != 'default.null':
        code.write(f'START_SND = "{snd}"\n')
    snd = logic_ent['snd_stop']
    if snd and snd.casefold() != 'default.null':
        code.write(f'STOP_SND = "{snd}"\n')
    snd = logic_ent['snd_move']
    if snd and snd.casefold() != 'default.null':
        logic_ent['classname'] = 'ambient_generic'
        logic_ent['message'] = snd
    else:
        logic_ent['classname'] = 'info_target'

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
    if (underside_fizz is not None or underside_hurt is not None
        or topside_fizz is not None or topside_hurt is not None
    ):
        if conv_bool(logic_ent['hurt_lenient']):
            logic_ent['thinkfunction'] = 'Think'
            code.write('\tfizz_eager = false\n')
        elif topside_fizz is not None or topside_hurt is not None:
            LOGGER.warning(
                'Piston platform {} is requesting topside fizzlers/hurts, '
                'but leinient hurting is required for this feature. Enabling.',
                ent_description(logic_ent),
            )
            logic_ent['thinkfunction'] = 'Think'
            code.write('\tfizz_eager = false\n')
        else:
            code.write('\tfizz_eager = true\n')

        def write_hurts(fizz: Entity | None, hurt: Entity | None, kind: str) -> None:
            """Write the hurt/fizzler entities into VScript."""
            name_fizz = fizz['targetname'] if fizz is not None else ''
            name_hurt = hurt['targetname'] if hurt is not None else ''

            if name_hurt:
                code.write(f'\tfizz_{kind}.player = Entities.FindByName(null, "{name_hurt}")\n')
            if name_fizz.casefold() == name_hurt.casefold() and name_hurt:
                # Both have the same name, we need to do some special handling.
                code.write(f'\tfizz_{kind}.obj = Entities.FindByName(fizz_{kind}.player, "{name_fizz}")\n')
                # If they're the wrong order, swap.
                # TODO: Could we swap their order in the entity lump to handle this?
                code.write(
                    f'\tif (fizz_{kind}.obj.GetClassname() == "trigger_hurt") {{\n'
                    f'\t\tlocal swap = fizz_{kind}.obj\n'
                    f'\t\tfizz_{kind}.obj = fizz_{kind}.player\n'
                    f'\t\tfizz_{kind}.player = swap\n'
                    '\t}\n'
                )
            elif name_fizz:
                code.write(f'\tfizz_{kind}.obj = Entities.FindByName(null, "{name_fizz}")\n')
            # Lenient fizzlers must always start disabled, VScript turns them on briefly only.
            if fizz is not None:
                fizz['startdisabled'] = True
            if hurt is not None:
                hurt['startdisabled'] = True

        write_hurts(underside_fizz, underside_hurt, 'dn')
        write_hurts(topside_fizz, topside_hurt, 'up')

    code.write('}\n')

    strip_cust_keys(logic_ent)
    logic_ent['vscripts'] = 'srctools/piston_platform.nut'
    ctx.add_code(logic_ent, code.getvalue())


def gen_logic_branches(
    *,
    ctx: Context, logic_ent: Entity,
    indices: range, pistons: dict[int, Piston],
    underside_ents: list[Entity | None],
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
    name = ''  # So the if below can check if the loop ran.
    for name in {ent['targetname'] for ent in underside_ents if ent is not None}:
        ctx.add_io_remap(
            logic_name,
            Output('Extend', name, 'Disable'),
            Output('Retract', name, 'Enable'),
        )
        pist_first.ent.add_out(Output(
            'OnFullyOpen' if pist_first.inverted else 'OnFullyClosed',
            name, 'Disable',
        ))
    if name and conv_bool(logic_ent['hurt_lenient']):
        LOGGER.warning(
            'Piston platform {} is requesting lenient underside fizzlers/hurts, '
            'but VScript must be enabled for this feature.',
            ent_description(logic_ent)
        )

    snd = logic_ent['snd_start']
    if snd and snd.casefold() != 'default.null':
        snd_ent = ctx.vmf.create_ent(
            'ambient_generic',
            origin=logic_ent['origin'],
            message=snd,
            sourceentityname=pist_last.ent['targetname'],
            radius=1500,
            spawnflags=16 | 32,  # Start Silent + Not looped.
        ).make_unique(logic_name + '_snd_start')
        ctx.add_io_remap(
            logic_name,
            Output('Extend', snd_ent, 'PlaySound'),
            Output('Retract', snd_ent, 'PlaySound'),
        )

    snd = logic_ent['snd_stop']
    if snd and snd.casefold() != 'default.null':
        snd_ent = ctx.vmf.create_ent(
            'ambient_generic',
            origin=logic_ent['origin'],
            message=snd,
            sourceentityname=pist_last.ent['targetname'],
            radius=1500,
            spawnflags=16 | 32,  # Start Silent + Not looped.
        ).make_unique(logic_name + '_snd_stop')
        # Only need to care about reaching the ends.
        pist_first.ent.add_out(Output(
            'OnFullyOpen' if pist_first.inverted else 'OnFullyClosed',
            snd_ent, 'PlaySound',
        ))
        pist_last.ent.add_out(Output(
            'OnFullyClosed' if pist_first.inverted else 'OnFullyOpen',
            snd_ent, 'PlaySound',
        ))

    snd = logic_ent['snd_move']
    if snd and snd.casefold() != 'default.null':
        snd_ent = ctx.vmf.create_ent(
            'ambient_generic',
            origin=Vec.from_str(logic_ent['origin']) + (0, 0, 16),
            message=snd,
            sourceentityname=pist_last.ent['targetname'],
            radius=1500,
            spawnflags=16,  # Start Silent.
        ).make_unique(logic_name + '_snd_move')
        ctx.add_io_remap(
            logic_name,
            Output('Extend', snd_ent, 'PlaySound'),
            Output('Retract', snd_ent, 'PlaySound'),
        )
        pist_first.ent.add_out(Output(
            'OnFullyOpen' if pist_first.inverted else 'OnFullyClosed',
            snd_ent, 'StopSound',
        ))
        pist_last.ent.add_out(Output(
            'OnFullyClosed' if pist_first.inverted else 'OnFullyOpen',
            snd_ent, 'StopSound',
        ))

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
