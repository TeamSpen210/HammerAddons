"""Transformations for packing and precaching resources."""
import os

from srctools import Entity
from srctools.logger import get_logger
from srctools.mdl import MDL_EXTS
from srctools.packlist import FileType, strip_extension, unify_path
from srctools.sndscript import SND_CHARS

from hammeraddons.bsp_transform import Context, check_control_enabled, trans


LOGGER = get_logger(__name__, 'trans.packing')


@trans('comp_precache_model', priority=100)
def comp_precache_model(ctx: Context) -> None:
    """Force precaching a specific model."""
    already_done: set[str] = set()
    for ent in ctx.vmf.by_class['comp_precache_model']:
        if not check_control_enabled(ent):
            ent.remove()
            continue
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


SND_CACHE_FUNC = '''\
function Precache() {
%s
}
'''


@trans('comp_precache_sound', priority=100)
def comp_precache_sound(ctx: Context) -> None:
    """Force precaching a set of sounds."""
    # Match normalised sound to the original filename.
    sounds: dict[str, str] = {}
    for ent in ctx.vmf.by_class['comp_precache_sound']:
        ent.remove()
        if not check_control_enabled(ent):
            continue

        for key, sound in ent.items():
            if not key.startswith('sound'):
                continue
            sound_key = sound.casefold().replace('\\', '/').lstrip(SND_CHARS)
            if sound_key.endswith(('.wav', '.mp3', '.ogg')) and not sound_key.startswith('sound/'):
                sound_key = 'sound/' + sound_key

            # Precaching implies packing it.
            ctx.pack.pack_file(sound_key, FileType.GAME_SOUND)

            sounds.setdefault(sound_key, sound)

    if not sounds:
        return

    # This VScript function forces a script to be precached.
    lines = SND_CACHE_FUNC % '\n'.join([
        f'\tself.PrecacheSoundScript("{snd.lstrip(SND_CHARS)}")'
        for snd in sorted(sounds.values())
    ])

    ctx.vmf.create_ent(
        'info_target',
        targetname='@precache',
        origin='-15872 -15872 -15872',  # Should be outside the map.
        vscripts=ctx.pack.inject_vscript(lines),
    )


@trans('comp_pack_replace_soundscript', priority=100)
def comp_pack_replace_soundscript(ctx: Context) -> None:
    """Replace a soundscript with a different one."""
    old_scripts = set()
    new_scripts = set()
    for ent in ctx.vmf.by_class['comp_pack_replace_soundscript']:
        ent.remove()
        if not check_control_enabled(ent):
            continue

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
def comp_pack(ctx: Context) -> None:
    """Force packing resources."""
    for ent in ctx.vmf.by_class['comp_pack']:
        ent.remove()
        if not check_control_enabled(ent):
            continue
        for key, value in ent.items():
            # Non-resource keyvalues.
            if key in {'classname', 'origin', 'angles', 'hammerid', 'skin', 'ctrl_type', 'ctrl_value'}:
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
def comp_pack_rename(ctx: Context) -> None:
    """Pack a file, under a different name."""

    # Optimisation, don't re-read files multiple times.
    # We're storing the data anyway.
    file_data: dict[str, bytes] = {}

    for ent in ctx.vmf.by_class['comp_pack_rename']:
        ent.remove()
        if not check_control_enabled(ent):
            continue

        name_src = unify_path(ent['filesrc'])
        name_dest = ent['filedest']
        file_type_name = ent['filetype']

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
            try:
                file = ctx.sys.open_bin(name_src)
            except FileNotFoundError:
                LOGGER.warning(
                    'File cannot be loaded to pack! \n{} -> {}',
                    name_src,
                    name_dest,
                )
                continue
            with file:
                data = file_data[name_src] = file.read()

        LOGGER.info('Force packing "{}" as "{}"...', name_src, name_dest)
        ctx.pack.pack_file(name_dest, res_type, data)

        if res_type is FileType.MODEL:
            # Pack additional files.
            name_src_stem = strip_extension(name_src)
            name_dest_stem = strip_extension(name_dest)
            for ext in MDL_EXTS:
                if ext == '.mdl':  # TODO use MDL_EXTS_EXTRA
                    continue

                name_add = name_src_stem + ext

                try:
                    data = file_data[name_add]
                except KeyError:
                    try:
                        file = ctx.sys.open_bin(name_add)
                    except FileNotFoundError:
                        # Optional.
                        continue
                    with file:
                        data = file_data[name_add] = file.read()

                LOGGER.info(
                    'Force packing "{}" as "{}{}"...',
                    name_add, name_dest_stem, ext,
                )
                ctx.pack.pack_file(name_dest_stem + ext, FileType.GENERIC, data)
