"""Implements various brush entities."""
from srctools.bsp_transform import trans, Context
from srctools import Output, conv_bool, conv_float


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
        for out in trig.outputs[:]:
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
            for out in manager.outputs[:]:
                if out.output.casefold() == 'onchangetoalltrue':
                    out.only_once = True
                else:
                    manager.outputs.remove(out)


@trans('P2 Goo')
def comp_trigger_goo(ctx: Context):
    """Creates triggers for Toxic Goo."""
    for trig in ctx.vmf.by_class['comp_trigger_p2_goo']:
        trig.remove()
        outputs = trig.outputs.copy()
        trig.outputs.clear()

        failsafe_delay = conv_float(trig['failsafe_delay'], 0.5)
        del trig['failsafe_delay']
        if failsafe_delay < 0.01:
            failsafe_delay = 0.01

        hurt = trig.copy()
        diss = trig.copy()
        ctx.vmf.add_ents([hurt, diss])

        hurt['classname'] = 'trigger_hurt'
        hurt['damagetype'] = 262144  # Radiation
        hurt['damage'] = hurt['damagecap'] = 10000
        hurt['damagemodel'] = 0  # No doubling
        hurt['nodmgforce'] = 1  # Don't throw players around.
        hurt['spawnflags'] = 1  # Players.
        del hurt['filtername']

        diss['classname'] = 'trigger_multiple'
        diss['spawnflags'] = 1096  # Physics, physics debris, everything
        diss['wait'] = 0  # No delay.
        diss['filtername'] = trig['dissolve_filter']

        diss.add_out(
            Output('OnStartTouch', '!activator', 'SilentDissolve'),
            Output('OnStartTouch', '!activator', 'Kill', delay=failsafe_delay),
        )

        for out in outputs:
            if out.output.casefold() == 'onkillplayer':
                out.output = 'OnStartTouch'
                hurt.add_out(out)
            elif out.output.casefold() == 'ondissolvephysics':
                out.output = 'OnStartTouch'
                diss.add_out(out)
