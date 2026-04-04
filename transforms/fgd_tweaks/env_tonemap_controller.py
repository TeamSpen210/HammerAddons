"""Allow setting all tonemap options on the controller, by generating inputs."""
from srctools.logger import get_logger
from srctools.vmf import Output

from hammeraddons.bsp_transform import trans, Context


# All the basic inputs.
INPUTS = [
    ("TonemapScale", "SetTonemapScale"),
    ("TonemapRate", "SetTonemapRate"),
    ("AutoExposureMin", "SetAutoExposureMin"),
    ("AutoExposureMax", "SetAutoExposureMax"),
    ("BloomExponent", "SetBloomExponent"),
    ("BloomSaturation", "SetBloomSaturation"),
    ("PercentBrightPixels", "SetTonemapPercentBrightPixels"),
    ("PercentBrightTarget", "SetTonemapPercentTarget"),
    ("MinAvgLum", "SetTonemapMinAvgLum"),
]
LOGGER = get_logger(__name__)


@trans('FGD - env_tonemap_controller inputs')
def env_tonemap_controller(ctx: Context) -> None:
    """Set tonemap options by generating inputs."""
    for tonemapper in ctx.vmf.by_class['env_tonemap_controller']:
        out = []
        # Might not be unique - if not, that's the user's problem.
        # Also don't bother checking validity, inputs can handle that.
        tone_name = tonemapper['targetname']
        for kv_name, inp_name in INPUTS:
            if (value := tonemapper[kv_name]) not in ('', '-1'):
                out.append(Output('OnMapSpawn', tone_name, inp_name, value))
        # Special case, bloomscale has a different input post-ASW with two options.
        # TODO: detect usage on older games.
        if (value := tonemapper['BloomScale']) not in ('', '-1'):
            out.append(Output(
                'OnMapSpawn', tone_name,
                'SetBloomScaleRange' if ' ' in value else 'SetBloomScale',
                value,
            ))
        if out:
            LOGGER.debug('Generating logic_auto to configure "{}" = {}', tone_name, out)
            ctx.vmf.create_ent(
                'logic_auto',
                origin=tonemapper['origin'],
                spawnflags='1',  # Remove on fire
            ).outputs = out
