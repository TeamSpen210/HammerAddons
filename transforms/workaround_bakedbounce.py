"""A workaround fix for named Baked Bounce lights flagging bounced lighting in lightmap pages."""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import VMF



LOGGER = get_logger(__name__)

@trans("workaround_bakedbounce", priority=-999999) # Run first
def bbounceworkaround(ctx: Context):
    vmf: VMF = ctx.vmf
    for light in vmf.by_class["light"] | vmf.by_class["light_spot"] | vmf.by_class["light_rt"] | vmf.by_class["light_rt_spot"]:
        if light["targetname", ""] and light["_lightmode", 2] == "2":
            light["_lightmode"] = 3
            light["style"] = 0
        

