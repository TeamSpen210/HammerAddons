"""Implements various brush entities."""
from typing import Tuple, Dict

from srctools.bsp_transform import trans, Context
from srctools import Output, conv_bool, conv_float, Vec, Entity, conv_int


@trans('P2 Coop Trigger')
def comp_trigger_coop(ctx: Context):
    """Creates a trigger which only activates with both players."""
    for trig in ctx.vmf.by_class['comp_trigger_coop']:
        trig['classname'] = 'trigger_playerteam'
        trig['target_team'] = 0
        
        only_once = conv_bool(trig['trigger_once'])
        trig['trigger_once'] = 0
        
        trig_name = trig['targetname']
        if not trig_name:
            # Give it something unique
            trig['targetname'] = trig_name = '_comp_trigger_coop_' + str(trig['hammer_id'])
            
        man_name = trig_name + '_man'
        
        manager = ctx.vmf.create_ent(
            classname='logic_coop_manager',
            origin=trig['origin'],
            targetname=man_name,
            # Should make it die if the trigger does.
            parentname=trig_name,
        )
        for out in list(trig.outputs):
            folded_out = out.output.casefold()
            if folded_out == 'onstarttouchboth':
                out.output = 'OnChangeToAllTrue'
            elif folded_out == 'onendtouchboth':
                out.output = 'OnChangeToAnyFalse'
            else:
                continue
            trig.outputs.remove(out)
            manager.add_out(out)
        trig.add_out(
            Output('OnStartTouchBluePlayer', man_name, 'SetStateATrue'),
            Output('OnStartTouchOrangePlayer', man_name, 'SetStateBrue'),
            Output('OnEndTouchBluePlayer', man_name, 'SetStateAFalse'),
            Output('OnEndTouchOrangePlayer', man_name, 'SetStateBFalse'),
        )
        
        if only_once:
            manager.add_out(
                Output('OnChangeToAllTrue', man_name, 'Kill'),
                Output('OnChangeToAllTrue', trig_name, 'Kill'),
            )
            # Only keep OnChangeToAllTrue outputs, and remove
            # them once they've fired.
            for out in list(manager):
                if out.output.casefold() == 'onchangetoalltrue':
                    out.only_once = True
                else:
                    manager.outputs.remove(out)


@trans('P2 Goo')
def comp_trigger_goo(ctx: Context):
    """Creates triggers for Toxic Goo."""
    reloader_cache = {}  # type: Dict[Tuple[float, float, float, float], Entity]

    for trig in ctx.vmf.by_class['comp_trigger_p2_goo']:
        brush_model = ctx.bsp.bmodels[trig]
        trig.remove()
        outputs = trig.outputs.copy()
        trig.outputs.clear()

        failsafe_delay = conv_float(trig['failsafe_delay'], 0.5)
        if failsafe_delay < 0.01:
            failsafe_delay = 0.01

        hurt = trig.copy()
        diss = trig.copy()
        ctx.vmf.add_ents([hurt, diss])
        spawnflags = conv_int(trig['spawnflags'])
        ctx.bsp.bmodels[hurt] = ctx.bsp.bmodels[diss] = brush_model

        for keyvalue in [
            'dissolve_filter',
            'phys_offset',
            'failsafe_delay',
            'fadepreset', 'fadecolor', 'fadetime',
        ]:
            del diss[keyvalue], hurt[keyvalue]

        diss['classname'] = 'trigger_multiple'
        # No clients, add physics. But otherwise leave it to the user.
        diss['spawnflags'] = (spawnflags & ~1) | 8
        diss['wait'] = 0  # No delay.
        diss['filtername'] = trig['dissolve_filter']
        del diss['damagetype']

        diss_pos = Vec.from_str(diss['origin'])
        diss_pos.z -= conv_float(trig['phys_offset'])
        diss['origin'] = diss_pos

        hurt['spawnflags'] = 1  # Players.

        if conv_bool(trig['enablefade']):
            fade_time = conv_float(trig['fadetime'])
            fade_color = Vec.from_str(trig['fadepreset'])
            if fade_color == (-1, -1, -1):
                fade_color = Vec.from_str(trig['fadecolor'])
            fade_key = fade_color.x, fade_color.y, fade_color.z, fade_time
            try:
                reloader = reloader_cache[fade_key]
            except KeyError:
                reloader = reloader_cache[fade_key] = ctx.vmf.create_ent(
                    'player_loadsaved',
                    origin=diss['origin'],
                    rendercolor=str(fade_color),
                    renderamt=255,
                    duration=fade_time,
                    holdtime=10,
                    loadtime=fade_time + 0.1,
                )
                reloader.make_unique('reloader')
            hurt['classname'] = 'trigger_once'
            del hurt['damagetype']
            hurt.add_out(Output('OnStartTouch', reloader, 'Reload', only_once=True))

            # Make sure the failsafe delay is longer than the total fade time.
            failsafe_delay = min(failsafe_delay, fade_time + 0.15)
        else:
            hurt['classname'] = 'trigger_hurt'
            hurt['damage'] = hurt['damagecap'] = 10000
            hurt['damagemodel'] = 0  # No doubling
            hurt['nodmgforce'] = 1  # Don't throw players around.

        for out in outputs:
            if out.output.casefold() == 'onkillplayer':
                # Better than OnStartTouch, doesn't apply for god mode.
                out.output = 'OnHurtPlayer'
                hurt.add_out(out)
            elif out.output.casefold() == 'ondissolvephysics':
                out.output = 'OnStartTouch'
                diss.add_out(out)

        diss.add_out(
            Output('OnStartTouch', '!activator', 'SilentDissolve'),
            Output('OnStartTouch', '!activator', 'Kill', delay=failsafe_delay),
        )
