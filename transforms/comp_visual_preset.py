"""It's a visual preset consisting of a setup of color_correction, env_fog_controller, and env_tonemap_controller.
"""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Entity, VMF, Output

LOGGER = get_logger(__name__)



@trans("comp_visual_preset")
def visual_preset(ctx: Context) -> None:
    """Sets a visual preset..."""
    vpreset: Entity
    vmf: VMF = ctx.vmf
    for vpreset in vmf.by_class["comp_visual_preset"]:
        vpreset.remove()
        LOGGER.log(0, f"Parsing visual preset: {vpreset['targetname']}")

        relay_ent = vmf.create_ent("logic_relay", 
                       targetname = "visual_preset_" + vpreset["targetname"],
                       angles = "0 0 0",
                       spawnflags = 0,
                       startdisabled = 0,
                       origin = vpreset.get_origin()
                       )
        
        
        ctx.add_io_remap( # Rebind the IO
                vpreset["targetname"],
                Output("Apply", relay_ent, "Trigger")
            )

        # Tonemapping
        tonemapper: Entity
        to_tm_outputs = []
        if (name := vpreset["tonemapper", None]) and (tonemappers := list(vmf.search(name))): # Check if we have set tonemapper, if it exists and set it as a variable for us to use
            if tonemappers:
                tonemapper = tonemappers[0]

            to_tm_outputs = [
                Output("OnTrigger", tonemapper, "SetAutoExposureMax", param=str(vpreset["tm_autoexposuremax", 2.0]) ),
                Output("OnTrigger", tonemapper, "SetAutoExposureMin", param=str(vpreset["tm_autoexposuremin", 0.5]) ),
                Output("OnTrigger", tonemapper, "SetBloomScale", param=str(vpreset["tm_bloomscale", 0.2]) ),
                Output("OnTrigger", tonemapper, "SetBloomExponent", param=str(vpreset["tm_bloomexponent", 2.2]) ),
            ]

        relay_ent.add_out(*to_tm_outputs)


        # Fog Controller

        LerpTo = "LerpTo" if vpreset["use_lerp", True] else ""

        fog_ent: Entity
        to_fog_outputs = []
        if (name := vpreset["fog_controller", None]) and (fog_ents := list(vmf.search(name))):
            if fog_ents:
                fog_ent = fog_ents[0]

            to_fog_outputs = [
                Output("OnTrigger", fog_ent, "SetColor"         + LerpTo, param=str(vpreset["fog_primary_color", 2.0]) ),
                Output("OnTrigger", fog_ent, "SetStartDist"     + LerpTo, param=str(vpreset["fog_start", 0.5]) ),
                Output("OnTrigger", fog_ent, "SetEndDist"       + LerpTo, param=str(vpreset["fog_end", 0.2]) ),
                Output("OnTrigger", fog_ent, "SetMaxDensity"    + LerpTo, param=str(vpreset["fog_max_density", 2.2]) ),
                Output("OnTrigger", fog_ent, "StartFogTransition", delay=0.05),
                Output("OnTrigger", fog_ent, "TurnOn", delay=0.02) # Ensure it's actually on.
            ]

        relay_ent.add_out(*to_fog_outputs)


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
                Output("OnTrigger", cc_ent, "Enable")
            )

            relay_ent.add_out(
                Output("OnUser1", cc_ent, "Disable")
            )

    # End loop

    presets = list(vmf.search("visual_preset_*")) # We strictly want a list, not a generator here
    for vpreset_relay in presets:
        us = vpreset_relay

        for other in presets:
            if us == other: # Ignore us
                continue

            us.add_out(
                Output("OnTrigger", other, "FireUser1") # Make sure that if we get enabled, we disable every other preset
                # This currently only disables the color correction, because every other value gets overriden anyways
            )

            
        
            

