"""Small tweaks to different entities to make them easier to use.


"""
import itertools
from typing import Dict, Optional, Set

from srctools import conv_int, Entity, Output
from srctools.logger import get_logger

from srctools.bsp_transform import trans, Context
from srctools.packlist import ALT_NAMES


LOGGER = get_logger(__name__)


@trans('trigger_brush Input Filters')
def trigger_brush_input_filters(ctx: Context) -> None:
    """Copy spawnflags on top of the keyvalue.

    This way you get checkboxes you can easily control.
    """
    for ent in ctx.vmf.by_class['trigger_brush']:
        if conv_int(ent['spawnflags']):
            ent['InputFilter'] = ent['spawnflags']


@trans('Fix alternate classnames')
def fix_alt_classnames(ctx: Context) -> None:
    """A bunch of entities have additional alternate names.

    Fix that by coalescing them all to one name.
    """
    for alt, replacement in ALT_NAMES.items():
        for ent in ctx.vmf.by_class[alt]:
            ent['classname'] = replacement


LISTENER_INPUTS = {
    'uniquestateon': ('SetValue', '1'),
    'uniquestateoff': ('SetValue', '0'),
    'uniquestateset': ('SetValue', None),
    'uniquestatetoggle': ('Toggle', ''),
}


@trans('Branch Listener UniqueState')
def branch_listener_unique_state(ctx: Context) -> None:
    """Add a set of new inputs to branch listeners.

    These generate a logic_branch unique to the specific ent.
    """
    # For each listener, specifies the smallest index we haven't checked.
    listener_ind = {}  # type: Dict[Entity, int]

    for ent in ctx.vmf.entities:
        branch: Optional[Entity] = None
        # Track which listeners the branch has been added to.
        cur_listeners: Set[Entity] = set()
        for out in ent.outputs[:]:
            try:
                inp_name, inp_parm = LISTENER_INPUTS[out.input.casefold()]
            except KeyError:  # Not our output.
                continue

            # Find listeners.
            keep_out = False
            found_listener = False
            for listener in list(ctx.vmf.search(out.target)):
                if listener['classname'].casefold() != 'logic_branch_listener':
                    # One of the ents isn't a listener, so keep the output.
                    # There might be some other unknown entity that uses these
                    # inputs.
                    keep_out = True
                    continue
                found_listener = True

                # Generate the branch if needed.
                if branch is None:
                    branch = ctx.vmf.create_ent(
                        'logic_branch',
                        origin=ent['origin'],
                        targetname=ent['targetname'] + '_branch',
                        initialvalue='0',
                    ).make_unique()

                # Add it to the listener if we haven't yet.
                if listener not in cur_listeners:
                    for i in range(listener_ind.get(listener, 1), 17):
                        if not listener['branch{:02}'.format(i)]:
                            listener['branch{:02}'.format(i)] = branch['targetname']
                            # Don't re-check stuff we've already done.
                            listener_ind[listener] = i + 1
                            break
                    else:
                        LOGGER.warning(
                            'logic_branch_listener "{}" has no spare slots, '
                            'but a UniqueState input was targetted at it!',
                            listener['targetname'],
                        )
                        continue
                    cur_listeners.add(listener)
                # And add in the outputs.
                ent.add_out(Output(
                    out.output,
                    branch,
                    inp_name,
                    inp_parm if inp_parm is not None else out.params,
                    out.delay,
                    times=out.times,
                    comma_sep=out.comma_sep,
                ))

            if found_listener and not keep_out:
                # Remove the output.
                ent.outputs.remove(out)
