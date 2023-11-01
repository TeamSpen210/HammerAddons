"""Team Fortress 2 specific transformations"""

from srctools import Output

from hammeraddons.bsp_transform import trans, Context

# Modified from the portal 2 indicator name transform, this might be able to be slightly cleaner
@trans('TF2 Control Point Props')
def comp_control_points(ctx: Context):
    """Adds key to TF2 Control Points to automatically set the skin of the base (or another prop)."""
    for ent in ctx.vmf.entities:
        prop_name = ent['src_propname']
        if not prop_name:
            continue

        ent['src_propname'] = ''

        ent.add_out(
            Output('OnCapTeam1' , prop_name, 'Skin', '1'),
            Output('OnCapTeam2', prop_name, 'Skin', '2'),
            Output('OnCapReset', prop_name, 'Skin', '0'),
        )