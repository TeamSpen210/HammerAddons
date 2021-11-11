"""Transformations for packing and precaching resources."""
import os
from typing import Set, Dict

from srctools import Entity
from srctools.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools.packlist import FileType, SoundScriptMode, unify_path


LOGGER = get_logger(__name__, 'trans.packing')


@trans('comp_precache_model', priority=100)
def comp_precache_model(ctx: Context):
    """Force precaching a specific model."""
    already_done: Set[str] = set()
    for ent in ctx.vmf.by_class['comp_precache_model']:
        model = ent['model']

        # Precaching implies packing it.
        skinset = {int(skin) for skin in ent['skinset'].split()}
        ctx.pack.pack_file(model, FileType.MODEL, skinset=skinset)

        if os.path.normcase(model) in already_done:
            ent.remove()
            continue
        already_done.add(os.path.normcase(model))
        make_precache_prop(ent)


def make_precache_prop(ent: Entity) -> None:
    """Alter this prop to act as a precache ent."""
    ent['classname'] = 'prop_dynamic_override'
    # Disable shadows and similar things on this to make it as cheap
    # as possible.
    ent['rendermode'] = '10'
    ent['disableshadowdepth'] = '1'
    ent['disableshadows'] = '1'
    ent['solid'] = '0'
    ent['shadowdepthnocache'] = '2'
    ent['spawnflags'] = '256'  # Start with collision off.
    ent['SuppressAnimSounds'] = '1'
    ent['DisableBoneFollowers'] = '1'  # Bone followers are extra ents, no thanks.
    ent['PerformanceMode'] = '2'  # "Full gibs on all platforms."
    ent['srctools_nopack'] = '1'  # Don't pack, since this doesn't have the skinset.

    # Move to a corner of the map, so it won't be in PVS generally.
    ent['origin'] = '-15872 -15872 -15872'


SND_CACHE_FUNC = b'''\
function Precache() {
%s
}
'''


@trans('comp_precache_sound', priority=100)
def comp_precache_sound(ctx: Context):
    """Force precaching a set of sounds."""
    sounds = set()
    for ent in ctx.vmf.by_class['comp_precache_sound']:
        ent.remove()

        for key, sound in ent.keys.items():
            if not key.startswith('sound'):
                continue
            sound = sound.casefold().replace('\\', '/')
            if sound.endswith(('.wav', '.mp3')) and not sound.startswith('sound/'):
                sound = 'sound/' + sound

            # Precaching implies packing it.
            ctx.pack.pack_file(sound, FileType.GAME_SOUND)

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
        origin='-15872 -15872 -15872',  # Should be outside the map.
        # We don't include scripts/vscripts in the filename.
        vscripts=ctx.pack.inject_file(lines, 'scripts/vscripts/inject', 'nut')[17:],
    )


@trans('comp_pack_replace_soundscript', priority=100)
def comp_pack_replace_soundscript(ctx: Context):
    """Replace a soundscript with a different one."""
    old_scripts = set()
    new_scripts = set()
    for ent in ctx.vmf.by_class['comp_pack_replace_soundscript']:
        ent.remove()
        old_scripts.add(ent['original', ''].casefold())
        new_scripts.add(ent['replacement', ''].casefold())

    old_scripts.discard('')
    new_scripts.discard('')
    # Old takes priority over new.
    new_scripts.difference_update(old_scripts)

    for script in old_scripts:
        ctx.pack.soundscript.force_exclude(script)
    for script in new_scripts:
        try:
            ctx.pack.load_soundscript(ctx.sys[script], always_include=True)
        except FileNotFoundError:
            LOGGER.warning('No soundscript file "{}"!', script)
        ctx.pack.pack_file(script, FileType.SOUNDSCRIPT)


# Keyvalue -> filetype.
PACK_TYPES = {
    'generic': FileType.GENERIC,
    'sound': FileType.GAME_SOUND,
    'model': FileType.MODEL,
    'material': FileType.MATERIAL,
    'particle': FileType.PARTICLE_FILE,
    'soundscript': FileType.SOUNDSCRIPT,
}


@trans('comp_pack', priority=100)
def comp_pack(ctx: Context):
    """Force packing resources."""
    for ent in ctx.vmf.by_class['comp_pack']:
        ent.remove()
        for key, value in ent.keys.items():
            # Not important.
            if key in {'classname', 'origin', 'angles', 'hammerid', 'skin'}:
                continue

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
            if res_type is FileType.SOUNDSCRIPT:
                try:
                    file = ctx.sys[value]
                except FileNotFoundError:
                    LOGGER.warning(
                        'Soundscript "{}" does not '
                        'exist (at {})',
                        value, ent['origin'],
                    )
                    continue
                # Force include the script, and then pack all sounds it used
                # since the user explicitly specified it.
                for sound in ctx.pack.load_soundscript(file, always_include=True):
                    LOGGER.info('Sound: {}', sound)
                    ctx.pack.pack_file(sound.name, FileType.GAME_SOUND)


@trans('comp_pack_rename', priority=100)
def comp_pack_rename(ctx: Context):
    """Pack a file, under a different name."""

    # Optimisation, don't re-read files multiple times.
    # We're storing the data anyway.
    file_data: Dict[str, bytes] = {}

    for ent in ctx.vmf.by_class['comp_pack_rename']:
        ent.remove()
        name_src = unify_path(ent['filesrc'])
        name_dest = ent['filedest']
        file_type_name = ent['filetype']

        try:
            file = ctx.sys[name_src]
        except FileNotFoundError:
            LOGGER.warning(
                'File cannot be loaded to pack! \n{} -> {}',
                name_src,
                name_dest,
            )
            continue

        try:
            res_type = PACK_TYPES[file_type_name.casefold()]
        except KeyError:
            LOGGER.warning(
                'Unknown resource type: "{}" @ {}',
                file_type_name,
                ent['origin'],
            )
            res_type = FileType.GENERIC

        try:
            data = file_data[name_src]
        except KeyError:
            with ctx.sys, ctx.sys.open_bin(file) as f:
                data = file_data[name_src] = f.read()

        ctx.pack.pack_file(name_dest, res_type, data)
