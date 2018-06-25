"""Portal-2 specific transformations."""
from srctools.bsp_transform import trans, Context
from srctools import Output, conv_bool


@trans('Fix Laser Catcher Skins')
def laser_catcher_skins(ctx: Context):
    """Fix Valve's bug where reloading saves causes lasers to get their skin wrong."""
    for ent in ctx.vmf.by_class['prop_laser_catcher']:
        if not conv_bool(ent['src_fix_skins'], True):
            continue

        deact_skin, act_skin = '23' if ent['SkinType'] == '1' else '01'

        # Look for outputs which do this already.
        name = ent['targetname']

        has_act = has_deact = False
        for out in ent.outputs:
            if has_act and has_deact:
                break
            if out.target == name or out.target == '!self':
                if out.input.casefold() == 'skin':
                    if out.params == act_skin:
                        has_act = True
                    elif out.params == act_skin:
                        has_act = True

        if not has_act:
            ent.add_out(Output('OnPowered', '!self', 'Skin', act_skin))
        if not has_deact:
            ent.add_out(Output('OnUnPowered', '!self', 'Skin', deact_skin))
