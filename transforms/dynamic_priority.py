from hammeraddons.bsp_transform import trans, Context

from srctools import VMF, Entity, conv_int, Vec, Output
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


@trans("Dynamic Priority")
def dynamic_priority(ctx: Context):
    vmf = ctx.vmf

    light: Entity

    lights = set(vmf.by_class["light_rt_spot"]) | set(vmf.by_class["light_rt"])

    added_logic = False

    all_lights = lights \
                | set(vmf.by_class["light"]) \
                | set(vmf.by_class["light_spot"]) \
                | set(vmf.by_class["light_environment"]) \
                | set(vmf.by_class["light_directional"])

    # Each switchable light has a lightstyle assigned to it, we have to check how many of them are used and then use the unused ones
    # Check https://github.com/ValveSoftware/source-sdk-2013/blob/39f6dde8fbc238727c020d13b05ecadd31bda4c0/src/utils/vbsp/writebsp.cpp#L982-L1038
    # for reference

    used_styles = set()

    for light in all_lights:
        used_styles.add(conv_int(light['style'], default=0))


    available_styles = set(range(32, 65 + 32))

    available_styles = list(available_styles - used_styles)
    
    lg0_static_style = available_styles[0]
    lg1_static_style = available_styles[1]
    lg0_dynamic_style = available_styles[2]
    lg1_dynamic_style = available_styles[3]
    
    for light in lights:

        if conv_int(light["_lightmode"], 2) == 1: # Convert specular to dynamic
            LOGGER.info(f"Converting light at {light.get_origin()} to Baked Bounce!")
            light["_lightmode"] = 2
        
        if conv_int(light["_lightmode"], 2) != 2: # Only Baked Bounce makes sense to have this functionality
            continue

        if light["targetname", ""] != "":
            LOGGER.info(f"Lights with targetnames will be skipped! Light: {light['targetname', '']} at {light.get_origin()}")
            continue

        dynpr = conv_int(light["_dynamic_priority"], default=-1)
        
        if dynpr not in (0, 1, 2):
            LOGGER.warning(f"Invalid _dynamic_priority for light at {light.get_origin()}, skipping...")
            continue


        if dynpr == 2: # On High, don't change since the light is always dynamic
            continue

        #LOGGER.info(f"Processing light at {light.get_origin()}")

        if not added_logic:
            added_logic = True
            LOGGER.info(f"Additionally, handler logic will be spawned at the position mentioned above!")
            AddLogic(vmf, light.get_origin())


        light_copy = light.copy()
        light_bounce = light.copy()
        #It turns out, bounce isn't needed, it will get generated from the dynamic light since it doesn't have the style kv

        light_bounce["_lightmode"] = 2 # Ensure Bounce is created
        #light_bounce["_removeaftercompile"] = 1 # Make VRAD remove this light after compilation
        # This trick allows us to create artificial bounce-only lights, because named lights don't get bounce lights
        
        # Okay it turns out this "trick" works only sometimes and sometimes it crashes the game
        
        light_bounce.add_out(
            Output("OnUser1", "!self", "Kill", "", 0.2)
        )

        # The thing is, even when switching the modes, bounce lights will remain on, because we're switching between groups and not on/off

        light["targetname"] = f"light_dynpr_dynamic_{dynpr}"

        #Dynamic lights don't need styles for networking
        if dynpr == 0:
            light["style"] = lg0_dynamic_style
        elif dynpr == 1:
            light["style"] = lg1_dynamic_style

        
        light_copy["targetname"] = f"light_dynpr_static_{dynpr}"

        if dynpr == 0:
            light_copy["style"] = lg0_static_style
        elif dynpr == 1:
            light_copy["style"] = lg1_static_style

        # Create a static copy
        light_copy["_lightmode"] = 0 # Fully static

        # We expect the mode to be medium by default, it also limits the amount of lights being switched at once when changing from this mode on map load
        #if dynpr == 1: # Medium, set the static light to dark
        #    spawnflags = conv_int(light_copy["spawnflags", 0])
        #    spawnflags |= 1 # Sets Initially Dark to True
        #    light_copy["spawnflags"] = spawnflags
        #
        #elif dynpr == 0: # Low, set the dynamic light to dark
        spawnflags = conv_int(light["spawnflags", 0])
        spawnflags |= 1 # Sets Initially Dark to True
        light["spawnflags"] = spawnflags

        vmf.add_ents([light_copy])
        vmf.add_ents([light_bounce])



def AddLogic(vmf: VMF, pos: Vec):
    """Add the necessary logic to load the saved state on every map load."""

    logic_auto = vmf.create_ent(
        classname = 'logic_auto',
        angles = Vec(0, 0, 0),
        spawnflags = 0,
        origin = pos
    )
    # Don't remove on fire, we can fire on every map load, even reloads. 
    # If the game saves the state of the lights the script won't do anything
    # and if it doesn't, this will make sure that they are switched to a correct mode

    logic_script = vmf.create_ent(
        classname = 'logic_script',
        angles = Vec(0, 0, 0),
        targetname = '@PC_dynpr',
        vscripts = 'dynamic_priority.nut',
        origin = pos
    )

    logic_auto.add_out(
        Output("OnMapSpawn", "@PC_dynpr", "RunScriptCode", "LoadFromMemory()"),
        Output("OnMapSpawn", "light_rt*", "FireUser1", only_once=True) # Kill bounce only lights
    )

    logic_script.add_out(
        Output("OnUser1", "!self", "RunScriptCode", "ChangeMode(1)"),
        Output("OnUser2", "!self", "RunScriptCode", "ChangeMode(0)"),
        Output("OnUser3", "!self", "RunScriptCode", "ChangeMode(2)")
    )

