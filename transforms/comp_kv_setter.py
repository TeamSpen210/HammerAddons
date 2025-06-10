"""Sets a keyvalue on an entity to a new value.

This is useful to compute spawnflags, or to adjust keyvalues when the target
entity's options can't be set to a fixup variable.
"""
import re

from srctools import Angle, conv_int, conv_bool, Vec, FGD
from srctools.fgd import EntityDef
from srctools.logger import get_logger

from hammeraddons.bsp_transform import trans, Context, ent_description
from hammeraddons.bsp_transform.common import check_control_enabled, get_multimode_value

LOGGER = get_logger(__name__)


@trans('comp_kv_setter')
def kv_setter(ctx: Context) -> None:
    """Sets a keyvalue on an entity to a new value."""
    for setter in ctx.vmf.by_class['comp_kv_setter']:
        setter.remove()
        if not check_control_enabled(setter):
            continue
        desc = ent_description(setter)

        mode = setter['mode', 'kv']
        if mode.casefold() == 'flags':
            is_flags = True
        elif mode.casefold() == 'kv':
            is_flags = False
        else:
            LOGGER.warning(
                'Unknown mode ({}) for {}!',
                mode,
                desc,
            )
            continue

        kv_name = setter['kv_name']
        kv_value = get_multimode_value(setter, prefix='kv_value_', desc='value')

        if conv_bool(setter['invert']):
            kv_value = '0' if conv_bool(kv_value) else '1'
        if conv_bool(setter['rotate']):
            pos = Vec.from_str(kv_value) @ Angle.from_str(setter['angles'])
            if conv_bool(setter['conv_ang']):  # Save converting back and forth.
                kv_value = str(pos.to_angle())
            else:
                kv_value = str(pos)
        elif conv_bool(setter['conv_ang']):
            kv_value = str(Vec.from_str(kv_value).to_angle())

        conf_flag_mask: str | int
        if is_flags:
            try:
                # Convert using Python's literal rules,
                # to allow binary/hex literals.
                conf_flag_mask = int(kv_name, base=0)
            except (TypeError, ValueError):
                conf_flag_mask = normalise_flag(kv_name)
            flag_enabled = conv_bool(kv_value)
        else:
            conf_flag_mask = 1
            flag_enabled = False
            kv_name = kv_name.strip()

        if not is_flags and not kv_name and not setter.outputs:
            # We have nothing to do?
            LOGGER.warning(
                '{} is set to do nothing at all. '
                'Provide a keyvalue to set, spawnflag to change or '
                'outputs to append.',
                desc,
            )

        found_ent = None
        flag_mask: int | None
        fgd_flags: dict[str, int | None] = {}

        for found_ent in ctx.vmf.search(setter['target']):
            if is_flags:
                spawnflags = conv_int(found_ent['spawnflags', '0'])
                if isinstance(conf_flag_mask, str):
                    # Try and lookup the FGD.
                    found_ent_cls = found_ent['classname']
                    try:
                        flag_mask = fgd_flags[found_ent_cls]
                    except KeyError:
                        flag_mask = fgd_flags[found_ent_cls] = get_fgd_mask(found_ent_cls, conf_flag_mask)
                        if flag_mask is None:
                            LOGGER.warning(
                                'Invalid spawnflags mask for {}!\n'
                                'Provide the name of a flag, a decimal integer, 0x45 hex or 0b01101 binary value.',
                                desc,
                            )
                            continue
                        LOGGER.info(
                            'FGD mask lookup: {0!r} in {1} = {2:#b} ({2})',
                            conf_flag_mask, found_ent_cls, flag_mask,
                        )
                    else:
                        if flag_mask is None:
                            continue  # Don't repeat an error.
                else:
                    flag_mask = conf_flag_mask
                if flag_enabled:
                    found_ent['spawnflags'] = spawnflags | flag_mask
                else:
                    found_ent['spawnflags'] = spawnflags & ~flag_mask
            elif kv_name:  # Don't set empty KVs...
                found_ent[kv_name] = kv_value

            for out in setter.outputs:
                found_ent.add_out(out.copy())

        if found_ent is None:
            LOGGER.warning(
                'No entities found named "{}" for {}', setter['target'], desc,
            )


def normalise_flag(flag: str) -> str:
    """Normalise the flag, so punctuation doesn't affect results."""
    flag = re.sub(r'[()]', '', flag.casefold())
    return re.sub(r'[ \t]+', ' ', flag)


def get_fgd_mask(ent_cls: str, search: str) -> int | None:
    try:
        ent_def = EntityDef.engine_def(ent_cls)
    except KeyError:
        LOGGER.warning('No FGD definition for entity {}!', ent_cls)
        return None
    try:
        flags_kv = ent_def.kv['spawnflags']
    except KeyError:
        LOGGER.warning('No spawnflags in FGD definition for entity {}!', ent_cls)
        return None
    for (mask, name, default, flags) in flags_kv.flags_list:
        if search == normalise_flag(name):
            return mask
    return None
