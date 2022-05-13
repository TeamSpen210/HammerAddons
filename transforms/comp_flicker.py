"""Fires on/off inputs repeatedly to simulate a flicker-on effect."""
import random

from srctools import Output, lerp, logger, conv_float
from srctools.bsp_transform import trans, Context

LOGGER = logger.get_logger(__name__)
INP_TURN_OFF = 'FireUser1'
OUT_TURN_OFF = 'OnUser1'
INP_TURN_ON = 'FireUser2'
OUT_TURN_ON = 'OnUser2'

INP_FLICK_OFF = 'FireUser3'
OUT_FLICK_OFF = 'OnUser3'
INP_FLICK_ON = 'FireUser4'
OUT_FLICK_ON = 'OnUser4'


@trans('comp_flicker')
def comp_flicker(ctx: Context) -> None:
    """When triggered, fires on/off inputs repeatedly to simulate a flicker-on effect."""
    for ent in ctx.vmf.by_class['comp_flicker']:
        ent['classname'] = 'info_target'
        ent_name = ent['targetname']

        total_time = conv_float(ent['total_time'], 1.5)
        flicker_min = max(conv_float(ent['flicker_min'], 0.05), 0.01)
        flicker_max = max(conv_float(ent['flicker_max'], 0.3), 0.01)
        variance = conv_float(ent['variance'])

        ctx.add_io_remap(
            ent_name,
            Output('TurnOff', ent_name, INP_TURN_OFF),
            Output('TurnOn',  ent_name, INP_TURN_ON),
            Output('FlickerOff', ent_name, INP_FLICK_OFF),
            Output('FlickerOn',  ent_name, INP_FLICK_ON),
        )

        for out in ent.outputs:
            out_name = out.output.casefold()
            if out_name.startswith('onuser'):
                LOGGER.warning(
                    'comp_flicker "{}" uses Fire/OnUserX outputs, which are being used by the '
                    "entity logic! It probably won't work properly.",
                    ent_name,
                )
            elif out_name == 'onturnedoff':
                out.output = OUT_TURN_OFF
            elif out_name == 'onturnedon':
                out.output = OUT_TURN_ON
            elif out_name == 'onflickeroffstart':
                out.output = OUT_FLICK_OFF
            elif out_name == 'onflickeronstart':
                out.output = OUT_FLICK_ON
            elif out_name == 'onflickeroffend':
                out.output = OUT_FLICK_OFF
                out.delay += total_time
            elif out_name == 'onflickeronend':
                out.output = OUT_FLICK_ON
                out.delay += total_time
            else:
                LOGGER.warning('Unknown comp_flicker output "{}" for "{}"', out.output, ent_name)

        mdl_name = ent['target_mdl']
        if mdl_name:
            ent.add_out(
                Output(OUT_TURN_ON, mdl_name, 'Skin', ent['mdl_skin_on']),
                Output(OUT_TURN_OFF, mdl_name, 'Skin', ent['mdl_skin_off']),
            )

        for out_name, start_state, min_point, max_point in [
            (OUT_FLICK_ON, False, 0.0, total_time),
            (OUT_FLICK_OFF, True, total_time, 0.0),
        ]:
            time = 0
            state = start_state
            while time < total_time:
                state = not state
                ent.add_out(Output(out_name, '!self', INP_TURN_ON if state else INP_TURN_OFF, delay=time))

                delay = lerp(time, min_point, max_point, flicker_min, flicker_max)
                time += delay + random.uniform(-variance, variance)

                delay = lerp(
                    time,
                    min_point, max_point,
                    flicker_min, flicker_max,
                )
                # Clamp to specified min/max.
                delay = min(flicker_max, max(flicker_min, delay)) + random.uniform(-variance, variance)
                # And enforce monotonicity.
                if delay < 0.01:
                    delay = 0.01

                time += delay

            # Force on exactly at the end time.
            ent.add_out(Output(out_name, '!self', INP_TURN_OFF if start_state else INP_TURN_ON, delay=time))
