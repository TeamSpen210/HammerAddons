"""Implements simple logic."""
from srctools import conv_bool
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger


LOGGER = get_logger(__name__)


@trans('comp_relay', priority=10)
def comp_relay(ctx: Context):
    """Implements comp_relay, allowing zero-overhead relay ents for managing outputs.

    These are collapsed into their callers.
    """
    # Output -> input that we convert.
    out_names = {
        'ontrigger': 'trigger',
        'onturnedon': 'turnon',
        'onturnedoff': 'turnoff',
    }
    # Add user outputs as well.
    for i in '12345678':
        out_names['onuser' + i] = 'fireuser' + i

    for relay in ctx.vmf.by_class['comp_relay']:
        # First, see if any entities exist with the same name that aren't
        # comp_relays. In that case, we need to keep the inputs.
        relay_name = relay['targetname']
        should_remove = not any(
            ent['classname'].casefold() != 'comp_relay'
            for ent in ctx.vmf.by_target[relay_name]
        )
        # If ctrl_type is 0, ctrl_value needs to be 1 to be enabled.
        # If ctrl_type is 1, ctrl_value needs to be 0 to be enabled.
        enabled = conv_bool(relay['ctrl_type']) != conv_bool(relay['ctrl_value'])
        for out in relay.outputs:
            try:
                inp_name = out_names[out.output.casefold()]
            except KeyError:
                LOGGER.warning(
                    'Unknown output "{}" on comp_relay "{}"!\n'
                    'This will be discarded.',
                    out.output, relay_name,
                )
                continue
            if enabled:
                out.output = inp_name
                ctx.add_io_remap(relay_name, out, remove=should_remove)
            elif should_remove:  # Still add a remap, to remove the outputs.
                ctx.add_io_remap_removal(relay_name, inp_name)
        relay.remove()
