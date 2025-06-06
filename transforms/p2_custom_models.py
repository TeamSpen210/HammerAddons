"""Add keyvalues to Portal 2 test element entities to automatically handle custom models."""

from srctools import conv_int

from hammeraddons.bsp_transform import trans, Context

SUPPORTED_ENTS = [
    "prop_weighted_cube",
    "prop_button",
    "prop_floor_cube_button",
    "prop_floor_ball_button",
    "prop_under_button",
    "prop_under_floor_button",
    "prop_wall_projector",
    "prop_glados_core",
    # TODO: needs extra VScript for the sprite, otherwise should work
    # "npc_security_camera",
    "npc_rocket_turret",
    "npc_personality_core",
    "prop_exploding_futbol",
    "prop_glass_futbol",
    "hot_potato",

    # Not supported:
    # prop_testchamber_door - reverses animations
    # prop_tractor_beam - breaks animations
    # prop_linked_portal_door - breaks animations
    # prop_monster_box - swaps models dynamically

    # Not tested:
    # prop_glass_futbol_spawner
    # hot_potato_spawner
    # prop_portal_stats_display
    # prop_rocket_tripwire
    # prop_telescopic_arm
    # prop_scaled_cube (Sixense)
    # prop_contraption_cube (Edu)
    # prop_contraption_cube_button (Edu)
]


@trans('Portal 2 Custom Models')
def p2_custom_models(ctx: Context) -> None:
    """Add keyvalues to Portal 2 test element entities to automatically handle custom models."""
    for classname in SUPPORTED_ENTS:
        for ent in ctx.vmf.by_class[classname]:
            model_type = conv_int(ent.pop('comp_custom_model_type'))

            if model_type == 0: # none
                continue
            elif model_type == 1: # script override
                cust_model = ent['model']
                # Make a comp_precache_model
                ctx.vmf.create_ent(
                    classname = 'comp_precache_model',
                    model = cust_model,
                )
                ctx.add_code(ent, 'function OnPostSpawn() { self.SetModel("' + cust_model + '") }')
            elif model_type == 2 and ent['classname'] == 'prop_weighted_cube': # cube type 6, for prop_weighted_cube only
                orig_cube_type = ent['CubeType']
                ent['CubeType'] = '6'
                # Revert to the original type on spawn
                ctx.add_code(
                    ent,
                    'function OnPostSpawn() { '
                    'EntFireByHandle(self, "AddOutput", '
                    f'"CubeType {orig_cube_type}", 0, self, self) }}'
                )
