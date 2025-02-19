"""Check for various forms of infinite recursion which will cause crashes/stack overflows ingame."""
from typing import cast
from graphlib import CycleError, TopologicalSorter
import sys

from srctools import Entity
from srctools.logger import get_logger

from hammeraddons.bsp_transform import Context, ent_description, trans


LOGGER = get_logger(__name__)


@trans('Entity parent loop check')
def parent_loop_check(ctx: Context) -> None:
    """Check for any loops in entity parents. This will cause an immediate crash on load.

    Since ents can only have one parent but we search by name, it might be possible to detect
    a cycle but at runtime only, depending on the exact order in the ent list. That's rather
    fragile and broken, better to error anyway.
    """
    graph: TopologicalSorter[Entity] = TopologicalSorter()
    for ent in ctx.vmf.entities:
        # Avoid calling graph.add() on blank parents. search() returns nothing, but these
        # will still fill the graph with useless nodes.
        if parent_name := ent['parentname']:
            # Since we search, we'll just ignore missing ents
            # - that's fine, we just can't check then.
            graph.add(ent, *ctx.vmf.search(parent_name))

    try:
        graph.prepare()
    except CycleError as exc:
        loop = cast('list[Entity]', exc.args[1])
        LOGGER.error(
            '\nCycle detected in parents - map will crash at runtime. Loop:\n{}',
            # Reverse so we show child -> parent direction since that's how you set these.
            '\n'.join(f'- {ent_description(ent)}' for ent in reversed(loop))
        )
        sys.exit(1)

# Fortunately, we only care about filters that can reference others.
REDIRECT_FILTERS = [
    'filter_blood_control', 'filter_damage_mod', 'filter_damage_transfer',
    'filter_redirect_inflictor', 'filter_redirect_owner', 'filter_redirect_weapon',
]


@trans('Filter cycle check')
def filter_loop_check(ctx: Context) -> None:
    """Check for any cycles in filters.

    This won't immediately crash, but it will whenever they're tested.
    That's entirely useless, so no point in not checking.
    """
    graph: TopologicalSorter[Entity] = TopologicalSorter()
    # filter_script could do anything, but obviously we can't check that.
    for clsname in REDIRECT_FILTERS:
        for ent in ctx.vmf.by_class[clsname]:
            graph.add(ent, *ctx.vmf.search(ent['damagefilter']))
    for ent in ctx.vmf.by_class['filter_multi']:
        for i in range(1, 21):
            graph.add(ent, *ctx.vmf.search(ent[f'filter{i:02}']))
    try:
        graph.prepare()
    except CycleError as exc:
        loop = cast('list[Entity]', exc.args[1])
        LOGGER.error(
            '\nCycle detected in filter entities - game will crash if these are tested. Loop:\n{}',
            # Reverse so we show child -> parent direction since that's how you set these.
            '\n'.join(f'-> {ent_description(ent)}' for ent in reversed(loop))
        )
        sys.exit(1)
