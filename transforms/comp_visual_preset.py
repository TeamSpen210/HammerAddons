"""It's a visual preset consisting of a setup of color_correction, env_fog_controller, and env_tonemap_controller.
"""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Entity, VMF, Output, conv_bool


LOGGER = get_logger(__name__)



@trans("comp_visual_preset")
def visual_preset(ctx: Context) -> None:
    """Sets a visual preset..."""
    vpreset: Entity
    vmf: VMF = ctx.vmf
    vpreset_list = []
    for vpreset in vmf.by_class["comp_visual_preset"]:
        vpreset.remove()
        LOGGER.debug("Parsing visual preset: {}", vpreset['targetname'])

        relay_ent = vmf.create_ent("logic_relay", 
                       targetname = vpreset["targetname"],

                       angles = "0 0 0",
                       spawnflags = 0,
                       startdisabled = 0,
                       origin = vpreset.get_origin()
                       )
        
        vpreset_list.append(relay_ent) # Add to the list so we can reference it later
        
        ctx.add_io_remap( # Rebind the IO
                vpreset["targetname"],
                Output("Apply", relay_ent, "Trigger")
            )

        # Tonemapping

        if tm_name := vpreset["tonemapper", None]: # Check if we have set the tonemapper, it doesn't matter if exists in the map, the IO then will have no receiver
            for _ in vmf.search(tm_name):
                relay_ent.add_out(
                    Output("OnTrigger", tm_name, "SetAutoExposureMax",  param=vpreset["tm_autoexposuremax", 2.0]),
                    Output("OnTrigger", tm_name, "SetAutoExposureMin",  param=vpreset["tm_autoexposuremin", 0.5]),
                    Output("OnTrigger", tm_name, "SetBloomScale",       param=vpreset["tm_bloomscale", 0.2]),
                    Output("OnTrigger", tm_name, "SetBloomExponent",    param=vpreset["tm_bloomexponent", 2.2]),
                )
                break



        # Fog Controller

        LerpTo = "LerpTo" if conv_bool(vpreset["use_lerp", True]) else ""

        if fog_ent := vpreset["fog_controller", None]:
            for _ in vmf.search(fog_ent): # Ensure one exists, else no need for us to create these outputs
                relay_ent.add_out(
                    Output("OnTrigger", fog_ent, "SetColor"         + LerpTo, param=vpreset["fog_primary_color", 2.0]),
                    Output("OnTrigger", fog_ent, "SetStartDist"     + LerpTo, param=vpreset["fog_start", 0.5]),
                    Output("OnTrigger", fog_ent, "SetEndDist"       + LerpTo, param=vpreset["fog_end", 0.2]),
                    Output("OnTrigger", fog_ent, "SetMaxDensity"    + LerpTo, param=vpreset["fog_max_density", 2.2]),
                    Output("OnTrigger", fog_ent, "StartFogTransition", delay=0.05),
                    Output("OnTrigger", fog_ent, "TurnOn", delay=0.02) # Ensure it's actually on.
                )
                break




        # Colorcorrection

        if (filename := vpreset["cc_filename", ""]): # We may not want to use CC
            cc_ent = vmf.create_ent("color_correction", 
                                    targetname = vpreset["targetname"] + "_colorcorrection",
                                    origin = vpreset.get_origin(),
                                    angles = "0 0 0",
                                    exclusive = "0",
                                    startdisabled = "1",

                                    fadeinduration = str(vpreset["cc_fadein", 1.0]),
                                    fadeoutduration = str(vpreset["cc_fadeout", 1.0]),

                                    maxfalloff = "-1",
                                    minfalloff = "-1",
                                    maxweight = "1.0",

                                    filename = f"materials/correction/{filename}.raw"
                                    )

            relay_ent.add_out(
                Output("OnTrigger", cc_ent, "Enable"),
                Output("OnUser1", cc_ent, "Disable")
            )

    # End loop


    for vpreset_relay in vpreset_list:
        us = vpreset_relay

        for other in vpreset_list:

            if us == other: # Ignore us
                continue

            us.add_out(
                Output("OnTrigger", other, "FireUser1") # Make sure that if we get enabled, we disable every other preset
                # This currently only disables the color correction, because every other value gets overriden anyways
            )

            
        
            

