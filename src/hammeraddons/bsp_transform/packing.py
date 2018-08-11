"""Transformations for packing and precaching resources."""
import os
from typing import Set

from srctools.bsp_transform import trans, Context
from srctools.packlist import FileType
from srctools.logger import get_logger

LOGGER = get_logger(__name__, 'trans.packing')


@trans('comp_precache_model')
def comp_precache_model(ctx: Context):
    """Force precaching a specific model."""
    already_done = set()  # type: Set[str]
    for ent in ctx.vmf.by_class['comp_precache_model']:
        model = ent['model']
        if os.path.normcase(model) in already_done:
            ent.remove()
            continue
        already_done.add(os.path.normcase(model))

        ent['classname'] = 'prop_dynamic_override'

        # Disable shadows and similar things on this to make it as cheap
        # as possible.
        ent['rendermode'] = '10'
        ent['disableshadowdepth'] = '1'
        ent['disableshadows'] = '1'
        ent['solid'] = '0'
        ent['shadowdepthnocache'] = '2'
        ent['spawnflags'] = '256'  # Start with collision off.

        # Move to a corner of the map, so it won't be in PVS generally.
        ent['origin'] = '-15872 -15872 -15872'

SND_CACHE_FUNC = b'''\
function Precache() {
%s
}
'''


@trans('comp_precache_sound')
def comp_precache_sound(ctx: Context):
    """Force precaching a set of sounds."""
    pos = '0 0 0'
    sounds = set()
    for ent in ctx.vmf.by_class['comp_precache_sound']:
        ent.remove()
        pos = ent['origin']

        for key, sound in ent.keys.items():
            if not key.startswith('sound'):
                continue
            sound = sound.casefold().replace('\\', '/')
            if sound.endswith(('.wav', '.mp3')) and not sound.startswith('sound/'):
                sound = 'sound/' + sound
            sounds.add(sound)

    if not sounds:
        return

    # This VScript function forces a script to be precached.
    lines = SND_CACHE_FUNC % b'\n'.join([
        b'\tself.PrecacheSoundScript("%s")' % snd.encode('utf8')
        for snd in sorted(sounds)
    ])

    ctx.vmf.create_ent(
        'info_target',
        targetname='@precache',
        origin='-1000 -1000 -1000',  # Should be outside the map.
        # We don't include scripts/vscripts
        vscripts=ctx.pack.inject_file(lines, 'scripts/vscripts/inject', 'nut')[17:],
    )

# Keyvalue -> filetype.
PACK_TYPES = {
    'generic': FileType.GENERIC,
    'sound': FileType.GAME_SOUND,
    'model': FileType.MODEL,
    'material': FileType.MATERIAL,
    'particle': FileType.PARTICLE_FILE,
}


@trans('comp_pack')
def comp_pack(ctx: Context):
    """Force packing resources."""
    for ent in ctx.vmf.by_class['comp_pack']:
        ent.remove()
        for key, value in ent.keys.items():  # type: str, str
            # We allow numeric suffixes for multiple - generic45.
            try:
                res_type = PACK_TYPES[key.rstrip('0123456789').casefold()]
            except KeyError:
                LOGGER.warning(
                    'Unknown resource type: "{}" @ {}',
                    key,
                    ent['origin'],
                )
                res_type = FileType.GENERIC
            ctx.pack.pack_file(value, res_type)
