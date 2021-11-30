"""Instance-related improvements."""
from typing import Dict, Optional, Tuple

from srctools import Entity
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger

LOGGER = get_logger(__name__)

# Number of OnProxyRelayX pairs we have, plus 1.
RELAY_MAX = 30 + 1


@trans('func_instance_io_proxy')
def io_proxy_tweaks(ctx: Context) -> None:
    """Tweak proxy relays to fix issues.

    This either collapses them entirely into callers, or keeps them and duplicates
    them when required to avoid using too many outputs. Collapsing eliminates the
    redundant entity use, but means you can't see the instance boundary in 'developer 2'
    displays or ent_fire the command yourself. It also can duplicate outputs, if many entities
    are triggering the instance.

    Options:
        * collapse: If set, remove func_instance_io_proxy entities from the map
    """
    if ctx.config.bool('collapse'):
        collapse_proxy_relays(ctx)
    else:
        duplicate_proxy_relays(ctx)


def collapse_proxy_relays(ctx: Context) -> None:
    """Collapse proxies into their callers."""
    for proxy in list(ctx.vmf.by_class['func_instance_io_proxy']):
        proxy_name = proxy['targetname']
        proxy.remove()
        for out in proxy.outputs:
            if out.output.casefold().startswith('onproxyrelay'):
                ctx.add_io_remap(proxy_name, out)
            else:
                LOGGER.warning(
                    '{}:{} is not a proxyrelay output?',
                    proxy_name,
                    out.output,
                )


def duplicate_proxy_relays(ctx: Context) -> None:
    """Duplicate proxies when required to allow infinite instance IO."""
    # Proxy name, ProxyRelayX -> new name, new index
    new_names: Dict[Tuple[str, int], Tuple[str, int]] = {}

    # First edit proxy outputs, then edit everything else.
    for orig_proxy in list(ctx.vmf.by_class['func_instance_io_proxy']):
        # Set to max, so the next will be generated immediately.
        cur_num = RELAY_MAX
        newest_proxy: Optional[Entity] = None
        orig_proxy.remove()  # Remove the original, add new ones.
        proxy_nums: Dict[int, Tuple[Entity, int]] = {}

        proxy_name = orig_proxy['targetname']

        for out in orig_proxy.outputs:
            if not out.output.casefold().startswith('onproxyrelay'):
                LOGGER.warning(
                    '{}:{} is not a proxyrelay output?',
                    proxy_name,
                    out.output,
                )
                continue
            if len(out.output) == 12:
                # Ignore/remove unused outputs.
                continue
            try:
                index = int(out.output[12:])
            except ValueError:
                LOGGER.warning(
                    '{}:{} has invalid proxyrelay number?',
                    proxy_name,
                    out.output,
                )
                continue

            # Have we already assigned the proxy for this?
            try:
                cur_proxy, new_index = proxy_nums[index]
            except KeyError:
                # We need to assign it to one.
                if cur_num == RELAY_MAX:
                    # We need a new proxy.
                    newest_proxy = ctx.vmf.create_ent(
                        'func_instance_io_proxy',
                        targetname=proxy_name,
                        origin=orig_proxy['origin']
                    ).make_unique()
                    new_index = cur_num = 1
                else:
                    new_index = cur_num
                cur_num += 1
                cur_proxy = newest_proxy
                proxy_nums[index] = cur_proxy, new_index

            out.output = 'OnProxyRelay{}'.format(new_index)
            cur_proxy.add_out(out)

            new_names[proxy_name, index] = cur_proxy['targetname'], new_index

    for ent in ctx.vmf.entities:
        for out in list(ent.outputs):
            # Input is OnProxyRelay as well. Why.
            if out.inst_in or not out.input.casefold().startswith('onproxyrelay'):
                continue

            if len(out.input) == 12:
                # Ignore/remove unused outputs.
                ent.outputs.remove(out)
                continue

            try:
                index = int(out.input[12:])
            except ValueError:
                LOGGER.warning(
                    '{}:{} has invalid proxyrelay number?',
                    ent['targetname'] or ent['classname'],
                    out.input,
                )
                continue

            try:
                out.target, new_index = new_names[out.target, index]
            except KeyError:
                LOGGER.warning('Unknown proxy "{}"?', out.target)
                continue

            out.input = 'OnProxyRelay{}'.format(new_index)
