"""Implement comp_choreo_sceneset."""
from srctools.bsp_transform import trans, Context
from srctools import Output, conv_bool, conv_float
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


@trans('comp_choreo_sceneset')
def sceneset(ctx: Context):
    """Chains a set of choreographed scenes together."""
    for ent in ctx.vmf.by_class['comp_choreo_sceneset']:
        scenes = [
            ent['scene{:02}'.format(i)]
            for i in range(1, 21)
            if ent['scene{:02}'.format(i)]
        ]
        if not scenes:
            LOGGER.warning(
                '"{}" at ({}) has no scenes!',
                ent['targetname'],
                ent['origin'],
            )
            continue

        if conv_bool(ent['play_dings']):
            scenes.insert(0, 'scenes/npc/glados_manual/ding_on.vcd')
            scenes.append('scenes/npc/glados_manual/ding_off.vcd')
        delay = conv_float(ent['delay'], 0.1)
        only_once = conv_bool(ent['only_once'])

        ent.remove()

        start_ent = None

        name = ent['targetname'] or '_choreo_{}'.format(ent.id)
        for i, scene in enumerate(scenes):
            part = ctx.vmf.create_ent(
                classname='logic_choreographed_scene',
                targetname=(
                    '{}_{}'.format(name, i)
                    if i > 0 else
                    name
                ),
                origin=ent['origin'],
                scenefile=scene,
            )
            if i + 1 < len(scenes):
                part.add_out(Output(
                    'OnCompletion',
                    '{}_{}'.format(name, i+1),
                    'Start',
                    delay=delay,
                ))
            if only_once:
                # When started blank the name so it can't be triggered,
                # then clean up after finished
                part.add_out(
                    Output('OnStart', '!self', 'AddOutput', 'targetname '),
                    Output('OnCompletion', '!self', 'Kill'),
                )
            if start_ent is None:
                start_ent = part

        assert start_ent is not None, "Has scenes but none made?"

        for out in ent.outputs:
            if out.output.casefold() == 'onstart':
                start_ent.add_out(out)
            elif out.output.casefold() == 'onfinish':
                # Part is the last in the loop.
                out.output = 'OnCompletion'
                part.add_out(out)
