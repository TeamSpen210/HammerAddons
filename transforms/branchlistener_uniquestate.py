"""Add a set of new inputs to branch listeners, which generate the branches automatically."""
from srctools import Entity, Output, logger

from hammeraddons.bsp_transform import trans, Context


LOGGER = logger.get_logger(__name__)


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
    listener_ind: dict[Entity, int] = {}

    for ent in ctx.vmf.entities:
        branch: Entity | None = None
        # Track which listeners the branch has been added to.
        cur_listeners: set[Entity] = set()
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
                        if not listener[f'branch{i:02}']:
                            listener[f'branch{i:02}'] = branch['targetname']
                            # Don't re-check stuff we've already done.
                            listener_ind[listener] = i + 1
                            break
                    else:
                        LOGGER.warning(
                            'logic_branch_listener "{}" used all 16 slots, '
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
