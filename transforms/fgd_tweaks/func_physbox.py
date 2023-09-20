"""Allow physboxes to completely override the mass of physics brushes."""
from srctools import conv_float, logger

from hammeraddons.bsp_transform import trans, Context
from srctools.bsp import BModel


LOGGER = logger.get_logger(__name__)


@trans('FGD - func_physbox mass override')
def mass_override(ctx: Context) -> None:
    """Allow overriding the mass entirely for physboxes."""
    for ent in iter(ctx.vmf.by_class['func_physbox']):
        override = ent.pop('ha_override_mass')
        if conv_float(override) <= 0.0:
            continue
        try:
            bmodel: BModel = ctx.bsp.bmodels[ent]
        except KeyError:
            LOGGER.warning(
                'func_physbox "{}" @ {} is not a brush?',
                ent['targetname'], ent['origin'],
            )
            continue
        if bmodel.phys_keyvalues is None:
            continue
        for solid in bmodel.phys_keyvalues.find_all('solid'):
            solid['mass'] = override
