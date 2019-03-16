"""Add "Indicator Name" keys to Portal 2 entities.

This generates env_texturetoggle entities which do the right thing.
If the one of the target entities is a prop_indicator_panel, it also
toggles that.
"""
from typing import List, Set

from srctools import Entity, Output
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger


LOGGER = get_logger(__name__)


# Entities with the keyvalue -> on, off output names.
IND_ENTS = {
    'prop_button': ('OnPressed', 'OnButtonReset'),
    'prop_under_button': ('OnPressed', 'OnButtonReset'),

    'prop_floor_button': ('OnPressed', 'OnUnPressed'),
    'prop_floor_cube_button': ('OnPressed', 'OnUnPressed'),
    'prop_floor_ball_button': ('OnPressed', 'OnUnPressed'),
    'prop_under_floor_button': ('OnPressed', 'OnUnPressed'),

    'prop_laser_catcher': ('OnPowered', 'OnUnPowered'),
    'prop_laser_relay': ('OnPowered', 'OnUnPowered'),
    'point_laser_target': ('OnPowered', 'OnUnPowered'),
}


@trans('P2 Antlines')
def comp_antlines(ctx: Context):
    """Adds indicator name keys to Portal 2 entities."""
    for ent in ctx.vmf.entities:
        try:
            out_on, out_off = IND_ENTS[ent['classname']]
        except KeyError:
            continue
        ind_name = ent['indicatorname']
        if not ind_name:
            continue

        ent['indicatorname'] = ''

        # These are the names, not the ents themselves.

        # Or brush ents holding overlays.
        ind_overlays = set()  # type: Set[str]
        ind_toggles = set()  # type: Set[str]
        # These need the right inputs.
        ind_panel_tim = set()  # type: Set[str]
        ind_panel_check = set()  # type: Set[str]

        # Panels without an indicator set - we can use
        # these instead of a texturetoggle.
        unused_panels = []  # type: List[Entity]

        for ind_ent in ctx.vmf.search(ind_name):
            cls = ind_ent['classname']
            if cls == 'info_overlay_accessor':
                ind_set = ind_overlays
            elif cls == 'prop_indicator_panel':
                if not ind_ent['indicatorlights']:
                    unused_panels.append(ind_ent)
                if ind_ent['istimer'] == '1':
                    ind_set = ind_panel_tim
                else:
                    ind_set = ind_panel_check
            elif cls == 'env_texturetoggle':
                ind_set = ind_toggles
            elif ind_ent['model'].startswith('*'):  # Brush model index
                ind_set = ind_overlays
            else:
                LOGGER.warning(
                    'Invalid indicator entity "{}" @ {}!',
                    ind_ent['targetname'],
                    ind_ent['origin'],
                )
                continue
            ind_set.add(ind_ent['targetname'].casefold())

        for ind_name in ind_overlays:
            if unused_panels:
                pan = unused_panels.pop()
                # Don't use a panel pointing back to itself.
                if pan['targetname'].casefold() != ind_name:
                    pan['indicatorlights'] = ind_name
                    continue
                else:
                    unused_panels.append(pan)

            toggle = ctx.vmf.create_ent(
                'env_texturetoggle',
                targetname=ent['targetname'] + '_toggle',
                target=ind_name,
                origin=ent['origin'],
            )
            toggle.make_unique()
            ind_toggles.add(toggle['targetname'])

        for ind_name in ind_toggles:
            ent.add_out(
                Output(out_on , ind_name, 'SetTextureIndex', '1'),
                Output(out_off, ind_name, 'SetTextureIndex', '0'),
            )

        for ind_name in ind_panel_check:
            ent.add_out(
                Output(out_on , ind_name, 'Check'),
                Output(out_off, ind_name, 'Uncheck'),
            )

        for ind_name in ind_panel_tim:
            ent.add_out(
                Output(out_on , ind_name, 'Start'),
                Output(out_off, ind_name, 'Reset'),
            )
