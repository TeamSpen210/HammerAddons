"""Small tweaks to different entities to make them easier to use.


"""
import itertools
from typing import Dict

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
    # For each listener, specifies the prefix we've chosen for the branches
    # we generate.
    base_names = {}  # type: Dict[Entity, str]
    # We clear this for every entity, it maps a listener to the branch we made.
    branches = {}  # type: Dict[Entity, Entity]

    # First, find a prefix that's unused by any entities.
    # Put assignment at the end so we use the unprefixed name if no one
    # else is using this name.
    group_prefix = '_br_unique'
    for group_ind in itertools.count(1):
        if not any(
            name.casefold().startswith(group_prefix)
            for name in
            ctx.vmf.by_target
            if name is not None
        ):
            break
        group_prefix = '_br_unique' + str(group_ind)

    listener_count = itertools.count()  # Unique names for each listener.

    for ent in ctx.vmf.entities:
        branches.clear()
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

                # First, we need to pick a group name for this listener,
                # and put it in the keyvalues.
                try:
                    branch_prefix = base_names[listener]
                except KeyError:
                    branch_prefix = base_names[listener] = '{}{}_'.format(group_prefix, next(listener_count))
                    for i in range(1, 17):
                        if not listener['branch{:02}'.format(i)]:
                            listener['branch{:02}'.format(i)] = branch_prefix + '*'
                            break
                    else:
                        LOGGER.warning(
                            'logic_branch_listener "{}" has no spare slots, '
                            'but a UniqueState input was targetted at it!',
                            listener['targetname'],
                        )
                        continue

                # Then, generate the branch if needed.
                try:
                    branch = branches[listener]
                except KeyError:
                    branches[listener] = branch = ctx.vmf.create_ent(
                        'logic_branch',
                        origin=ent['origin'],
                        initialvalue='0',
                    ).make_unique(branch_prefix)
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
