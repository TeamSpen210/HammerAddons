"""Implement comp_choreo_sceneset."""
from srctools import Entity, Output, conv_bool, conv_float
from srctools.logger import get_logger

from hammeraddons.bsp_transform import trans, Context

LOGGER = get_logger(__name__)


@trans('comp_choreo_sceneset')
def sceneset(ctx: Context) -> None:
    """Chains a set of choreographed scenes together."""
    ent: Entity
    for ent in ctx.vmf.by_class['comp_choreo_sceneset']:
        scenes = [
            ent[f'scene{i:02}']
            for i in range(1, 21)
            if ent[f'scene{i:02}']
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

        scene_ents: list[Entity] = []

        name = ent.make_unique('_choreo')['targetname']
        for i, scene in enumerate(scenes):
            part = ctx.vmf.create_ent(
                classname='logic_choreographed_scene',
                targetname=(
                    f'{name}_{i}'
                    if i > 0 else
                    name
                ),
                origin=ent['origin'],
                scenefile=scene,
            )
            scene_ents.append(part)
            if i + 1 < len(scenes):
                part.add_out(Output(
                    'OnCompletion',
                    f'{name}_{i + 1}',
                    'Start',
                    delay=delay,
                ))
            if only_once:
                # When started blank the name so that it can't be triggered, then clean up after
                # it's finished
                part.add_out(
                    Output('OnStart', '!self', 'AddOutput', 'targetname '),
                    Output('OnCompletion', '!self', 'Kill'),
                )

        for out in ent.outputs:
            if out.output.casefold() == 'onstart':
                scene_ents[0].add_out(out)
            elif out.output.casefold() == 'onfinish':
                out.output = 'OnCompletion'
                scene_ents[-1].add_out(out)

        # Make firing cancel at the first scene also cancel the others.
        ctx.add_io_remap(name, *[
            Output('Cancel', ent, 'Cancel')
            for ent in scene_ents[1:]
        ], remove=False)
