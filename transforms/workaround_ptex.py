"""A workaround fix for ptexes, see: https://github.com/StrataSource/Engine/issues/1465"""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Entity, VMF, Output, conv_int



LOGGER = get_logger(__name__)

@trans("workaround_ptex")
def ptexworkaround(ctx: Context):
    vmf: VMF = ctx.vmf

    logic_auto = Entity(
        vmf, {
            "classname": "logic_auto",
            "spawnflags": "1",
        }
    )

    vmf.add_ent(logic_auto)

    i = 1

    for ptex in vmf.by_class["env_projectedtexture"]:
        flags = conv_int(ptex["spawnflags"], 1)
        if flags % 2 == 1: # Checks first flag, "Enabled"
            continue

        if not ptex["targetname", ""]:
            ptex["targetname"] = f"PTEX_{i}"
            i += 1

        LOGGER.info(f"Patching {ptex["targetname"]}!")

        logic_auto.add_out(
            Output("OnMapSpawn", ptex, "TurnOn", "", 0.1, only_once=True),
            Output("OnMapSpawn", ptex, "TurnOff", "", 1, only_once=True)
        )
        

