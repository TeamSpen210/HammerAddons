"""Add a keyvalue to prop_weighted_cube to automatically handle custom models."""

from srctools import conv_int

from hammeraddons.bsp_transform import trans, Context

@trans('Easy Custom Cube Models')
def custom_cube_models(ctx: Context) -> None:
    """Add a keyvalue to prop_weighted_cube to automatically handle custom models."""
    for ent in ctx.vmf.by_class['prop_weighted_cube']:
        model_type = conv_int(ent['comp_custom_model_type'])

        if model_type == 0: # none
            continue
        elif model_type == 1: # script override
            cube_model = ent['model']
            # Make a comp_precache_model
            ctx.vmf.create_ent(
                classname = 'comp_precache_model',
                model = cube_model,
            )
            ctx.add_code(ent, 'function OnPostSpawn() { self.SetModel("' + cube_model + '") }')
        elif model_type == 2: # cube type 6
            orig_cube_type = ent['CubeType']
            ent['CubeType'] = '6'
            # Revert to the original type on spawn
            ctx.add_code(ent, 'function OnPostSpawn() { EntFireByHandle(self, "AddOutput", "CubeType ' + orig_cube_type + '", 0, self, self) }')
        