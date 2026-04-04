"""Add outputs to Portal floor buttons."""
from hammeraddons.bsp_transform import trans, Context, ent_description
from srctools import Output
from srctools.logger import get_logger


LOGGER = get_logger(__name__)


@trans("P2 Floor Buttons")
def floor_buttons(ctx: Context) -> None:
    """Add additional outputs to Portal floor buttons."""
    for clsname in [
        'prop_floor_button', 'prop_floor_cube_button', 'prop_floor_ball_button',
        'prop_under_floor_button', 'prop_contraption_cube_button',
    ]:
        for button in ctx.vmf.by_class[clsname]:
            filter_outs = []
            other_outs = []
            for output in button.outputs[:]:
                match output.output.casefold():
                    case 'onpressedplayer':
                        output.output = 'OnPass'
                        filter_outs.append(output)
                    case 'onpressedcube':
                        output.output = 'OnFail'
                        filter_outs.append(output)
                    case _:
                        other_outs.append(output)
            button.outputs = other_outs
            if not filter_outs:
                continue
            filter_ent = ctx.vmf.create_ent(
                'filter_activator_class',
                targetname=f'{button["targetname"] or "button"}_filter',
                origin=button['origin'],
                filterclass='player',
                negated=0,
            )
            filter_ent.make_unique()
            filter_ent.outputs = filter_outs
            button.outputs.append(Output('OnPressed', filter_ent,'TestActivator'))
            LOGGER.debug(
                'Adding filter: "{}" -> "{}"',
                ent_description(button), filter_ent['targetname'],
            )
