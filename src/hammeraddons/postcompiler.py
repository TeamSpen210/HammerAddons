"""Runs before VRAD, to run operations on the final BSP."""
# ruff: noqa: E402 - allow non-toplevel imports here
from pathlib import Path
import sys
import warnings

from srctools.logger import Formatter, init_logging
import trio


# Put the logs in the executable folders.
LOGGER = init_logging(Path(sys.argv[0]).with_name('postcompiler.log'))
warnings.filterwarnings(category=DeprecationWarning, module='srctools', action='once')

from collections import defaultdict
from logging import FileHandler, StreamHandler, shutdown as logging_shutdown
import argparse
import math
import os
import re
import shutil

from srctools.bsp import BSP, BSP_LUMPS
from srctools.filesys import ZipFileSystem
from srctools.packlist import PackList

from hammeraddons import (
    BINS_PATH, HADDONS_VER, SRCTOOLS_VER, config, mdl_compiler, propcombine,
)
from hammeraddons.bsp_transform import run_transformations
from hammeraddons.move_shim import install as install_depmodule_hook


install_depmodule_hook()

DEBUG_LUMPS = False


def format_bytesize(val: float) -> str:
    """Add mb, gb etc suffixes to a size in bytes."""
    if val < 1024:
        return f'{val} bytes'  # No rounding.
    val /= 1024.0
    for size in ['kB', 'mB', 'gB']:
        if val <= 1024.0:
            return f'{val:.3f}{size}'
        val /= 1024.0
    return f'{val:.03f}tB'


async def main(argv: list[str]) -> None:
    """Run the postcompiler."""
    LOGGER.info('HammerAddons postcompiler, srctools=v{}, addons=v{}', SRCTOOLS_VER, HADDONS_VER)

    parser = argparse.ArgumentParser(
        description="Modifies the BSP file, allowing additional entities "
                    "and bugfixes.",
    )

    parser.add_argument(
        "--config",
        dest="config",
        default="",
        help="Specify the location of hammeraddons.vdf, overriding calculation from the map path.",
    )

    parser.add_argument(
        "-game", "--game",
        dest="game_folder",
        default="",
        help="Specify the folder containing gameinfo.txt, and thus the "
             "location of the game. This overrides the option specified "
             "in hammeraddons.vdf.",
    )

    parser.add_argument(
        "--nopack",
        dest="allow_pack",
        action="store_false",
        help="Prevent packing of files found in the map."
    )
    parser.add_argument(
        "--nosaving",
        dest="allow_save",
        action="store_false",
        help="For testing purposes, allow skipping saving the BSP.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force models and similar resources to be regnerated.",
    )
    parser.add_argument(
        '-v', '--verbose',
        action="store_true",
        help="Show DEBUG level messages.",
    )
    parser.add_argument(
        "--propcombine",
        action="store_true",
        help="Allow merging static props together.",
    )
    parser.add_argument(
        "--showgroups",
        action="store_true",
        help="Legacy option, use r_colorstaticprops ingame.",
    )
    parser.add_argument(
        "--dumpgroups",
        action="store_true",
        help="Write all props without propcombine groups to a new VMF.",
    )

    parser.add_argument(
        "map",
        help="The path to the BSP file.",
    )

    args = parser.parse_args(argv)

    if args.showgroups:
        LOGGER.warning('--showgroups is not implemented. r_colorstaticprops does the same thing ingame.')
    if args.verbose:
        # Find the stdout handler, make it DEBUG mode.
        for handler in LOGGER.handlers:
            if isinstance(handler, StreamHandler) and handler.stream is sys.stdout:
                handler.setLevel('DEBUG')
                break
        else:
            LOGGER.warning('Could not set stdout handler to DEBUG mode.')

    mdl_compiler.force_regen = args.regenerate

    # The path is the last argument to the compiler.
    # Hammer adds wrong slashes sometimes, so fix that.
    # Also, if it's the VMF file, make it the BSP.
    path = Path(args.map).with_suffix('.bsp')

    # Open and start writing to the map's log file.
    handler = FileHandler(path.with_suffix('.log'))
    handler.setFormatter(Formatter(
        # One letter for level name
        '[{levelname}] {module}.{funcName}(): {message}',
        style='{',
    ))
    LOGGER.addHandler(handler)
    LOGGER.info("Map path is {}", path)

    conf = config.parse(path, args.config, args.game_folder)

    packlist = PackList(conf.fsys)
    studiomdl_path = conf.game_conf.resolve_studiomdl(conf.expand_path)

    LOGGER.info('Gameinfo: {}', conf.game.path)
    LOGGER.info(
        'Search paths: \n{}',
        '\n'.join([system.path for system, prefix in conf.fsys.systems]),
    )

    LOGGER.info('Loading soundscripts...')
    assert conf.opts.path is not None
    packlist.load_soundscript_manifest(conf.loc.with_name('srctools_sndscript_data.dmx'))
    LOGGER.info('Done! ({} sounds)', len(packlist.soundscript))
    LOGGER.info('Loading particles...')
    packlist.load_particle_manifest(conf.loc.with_name('srctools_particle_data.dmx'))
    LOGGER.info('Done! ({} particles)', len(packlist.particles))
    LOGGER.info('Loading choreo scenes...')
    try:
        packlist.load_choreo_scenes()
    except FileNotFoundError:
        LOGGER.warning('No scenes.image file, scenes will not be loaded.')
    else:
        LOGGER.info('Done! ({} scenes)', len(packlist.choreo))

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)

    orig_lumps: dict[BSP_LUMPS | bytes, bytes] = {}
    debug_lump_folder = path.parent / f'{path.stem}_lumps'
    if DEBUG_LUMPS:
        debug_lump_folder.mkdir(exist_ok=True)
        lump_id: BSP_LUMPS | bytes
        for lump_id in BSP_LUMPS:
            orig_lumps[lump_id] = bsp_file.lumps[lump_id].data
            (debug_lump_folder / f'{lump_id.name}_old.lmp').write_bytes(orig_lumps[lump_id])
        for lump_id, game_lump in bsp_file.game_lumps.items():
            LOGGER.info('Lump {} = v{}', lump_id, game_lump.version)
            (debug_lump_folder / f'game_{lump_id.hex()}_old.lmp').write_bytes(game_lump.data)
            orig_lumps[lump_id] = game_lump.data

    # Mount the existing packfile, so the cubemap files are recognised.
    LOGGER.info('Mounting BSP packfile...')
    conf.fsys.add_sys(ZipFileSystem('<BSP pakfile>', bsp_file.pakfile))

    modelcompile_dump_str = conf.opts.get(config.MODEL_COMPILE_DUMP)
    modelcompile_dump = conf.expand_path(modelcompile_dump_str) / path.stem if modelcompile_dump_str else None
    if modelcompile_dump is not None:
        LOGGER.info('Clearing model compile dump folder {}', modelcompile_dump)
        modelcompile_dump.mkdir(parents=True, exist_ok=True)
        try:
            for file in modelcompile_dump.iterdir():
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()
        except FileNotFoundError:
            pass  # Already empty.

    bsp_file.out_comma_sep = conf.game_conf.io_comma_sep

    LOGGER.info('Running transforms...')
    await run_transformations(
        bsp_file.ents,
        conf.fsys, packlist,
        bsp_file,
        conf.game,
        conf.game_conf,
        disabled={name.strip().casefold() for name in conf.opts.get(config.DISABLED_TRANSFORMS).split(',')},
        modelcompile_dump=modelcompile_dump,
        studiomdl_path=studiomdl_path,
    )

    if args.propcombine:
        if studiomdl_path is None:
            # All builtin definitions should have a correct path, so this should be a user one.
            raise ValueError(
                'Cannot propcombine, no studioMDL path provided! '
                'Update hammeraddons_game.vdf.'
            )
        decomp_cache_path = conf.opts.get(config.PROPCOMBINE_CACHE)
        decomp_cache_loc: Path | None
        crowbar_loc: Path | None
        if decomp_cache_path is not None:
            decomp_cache_loc = conf.expand_path(decomp_cache_path)
            decomp_cache_loc.mkdir(parents=True, exist_ok=True)
        else:
            decomp_cache_loc = None
        if conf.opts.get(config.PROPCOMBINE_CROWBAR):
            # argv[0] is the location of our script/exe, which lets us locate
            # Crowbar from there. The environment var is for testing.
            if 'CROWBAR_LOC' in os.environ:
                crowbar_loc = Path(os.environ['CROWBAR_LOC']).resolve()
            else:
                crowbar_loc = Path(BINS_PATH, 'Crowbar.exe').resolve()
        else:
            crowbar_loc = None

        LOGGER.info('Combining props...')
        max_auto_range: float | None = conf.opts.get(config.PROPCOMBINE_MAX_AUTO_RANGE)
        if not max_auto_range:
            max_auto_range = math.inf
        await propcombine.combine(
            bsp_file,
            bsp_file.ents,
            packlist,
            conf.game,
            studiomdl_path,
            qc_folders=conf.opts.get(config.PROPCOMBINE_QC_FOLDER).as_array(conv=conf.expand_path),
            decomp_cache_loc=decomp_cache_loc,
            crowbar_loc=crowbar_loc,
            min_auto_range=conf.opts.get(config.PROPCOMBINE_MIN_AUTO_RANGE),
            max_auto_range=max_auto_range,
            min_cluster=conf.opts.get(config.PROPCOMBINE_MIN_CLUSTER),
            min_cluster_auto=conf.opts.get(config.PROPCOMBINE_MIN_CLUSTER_AUTO),
            blacklist=conf.opts.get(config.PROPCOMBINE_BLACKLIST).as_array(),
            volume_tolerance=conf.opts.get(config.PROPCOMBINE_VOLUME_TOLERANCE),
            compile_dump=modelcompile_dump,
            debug_dump=args.dumpgroups,
            pack_models=conf.opts.get(config.PROPCOMBINE_PACK) or False,
        )
        LOGGER.info('Done!')

    # Always strip the propcombine entities, since we don't need them either way.
    could_propcombine = conf.opts.get(config.PROPCOMBINE_MIN_AUTO_RANGE) is not None
    for ent in bsp_file.ents.by_class['comp_propcombine_set']:
        ent.remove()
        could_propcombine = True
    for ent in bsp_file.ents.by_class['comp_propcombine_volume']:
        bsp_file.bmodels.pop(ent, None)  # Ignore if not present.
        ent.remove()
        could_propcombine = True

    # Warn if propcombine was enabled by config but not by command line.
    if could_propcombine and not args.propcombine:
        LOGGER.warning('No propcombine allowed, --propcombine not passed on the command line!')

    auto_pack = conf.opts.get(config.AUTO_PACK)
    if auto_pack and args.allow_pack:
        LOGGER.info('Analysing packable resources...')
        packlist.pack_from_ents(
            bsp_file.ents,
            mapname=Path(bsp_file.filename).stem,  # TODO: Include directories?
            tags=conf.game_conf.tags,
        )

        packlist.pack_from_bsp(bsp_file)

        packlist.eval_dependencies()
        if conf.opts.get(config.SOUNDSCRIPT_MANIFEST):
            packlist.write_soundscript_manifest()
        if conf.opts.get(config.PARTICLES_MANIFEST):
            man_name = conf.game_conf.particles_manifest.replace('<map name>', path.stem)
            if not man_name:
                raise ValueError(
                    'Packing particle manifest requested, but filename not known. '
                    'Update hammeraddons_game.vdf, or report as a bug!'
                )
            LOGGER.info('Writing particle manifest "{}"...', man_name)
            packlist.write_particles_manifest(man_name)
        fsys_blocklist = conf.pack_blacklist
    else:
        # Disallow all systems. We still execute the packing logic to include generated files.
        fsys_blocklist = {
            sys for sys, _ in packlist.fsys.systems
            if sys.path != '<BSP pakfile>'
        }

    if auto_pack and not args.allow_pack:
        # Warn if packing was enabled by config but not by command line.
        # May be intentional, but can be confusing.
        LOGGER.warning('--nopack passed, packing has been disabled!')

    pack_allowlist = list(config.packfile_filters(conf.opts.get(config.PACK_ALLOWLIST), 'allowlist'))
    pack_blocklist = list(config.packfile_filters(conf.opts.get(config.PACK_BLOCKLIST), 'blocklist'))

    if conf.opts.get(config.PACK_STRIP_CUBEMAPS):
        pack_blocklist.append(re.compile(config.CUBEMAP_REGEX))

    LOGGER.debug('Packing allowlist={}, blocklist={}', pack_allowlist, pack_blocklist)

    def pack_callback(path: str) -> bool | None:
        """Check the file against the two lists."""
        for pattern in pack_allowlist:
            if pattern.search(path) is not None:
                return True
        for pattern in pack_blocklist:
            if pattern.search(path) is not None:
                return False
        return None

    dump_path = conf.opts.get(config.PACK_DUMP)
    if dump_path:
        packlist.pack_into_zip(
            bsp_file,
            blacklist=fsys_blocklist,
            ignore_vpk=False,
            callback=pack_callback,
            dump_loc=conf.expand_path(dump_path.lstrip('#')).absolute().resolve(),
            only_dump=dump_path.startswith('#'),
        )
    else:
        packlist.pack_into_zip(
            bsp_file,
            blacklist=fsys_blocklist,
            ignore_vpk=False,
            callback=pack_callback,
        )

    # List out all the files, but group together files with the same extension.
    ext_for_name: dict[str, list[str]] = defaultdict(list)
    for zip_info in bsp_file.pakfile.infolist():
        filename = Path(zip_info.filename)
        if '.' in filename.name:
            stem, ext = filename.name.split('.', 1)
            file_path = str(filename.parent / stem)
        else:
            file_path = zip_info.filename
            ext = ''

        ext_for_name[file_path].append(ext)

    LOGGER.info('Packed files: \n{}'.format('\n'.join([
        (
            f'{name}.{exts[0]}'
            if len(exts) == 1 else
            f'{name}.({"/".join(exts)})')
        for name, exts in sorted(ext_for_name.items())
    ])))

    if args.allow_save:
        LOGGER.info('Writing BSP...')
        bsp_file.save()

    LOGGER.info('Packfile size: {}', format_bytesize(
        len(bsp_file.lumps[BSP_LUMPS.PAKFILE].data)
    ))

    if DEBUG_LUMPS:
        LOGGER.info('Changed lumps:')
        for lump_id in BSP_LUMPS:
            data = bsp_file.lumps[lump_id].data
            (debug_lump_folder / f'{lump_id.name}_new.lmp').write_bytes(data)
            if orig_lumps.pop(lump_id) != data:
                LOGGER.info('{} changed', lump_id)
        for lump_id, game_lump in bsp_file.game_lumps.items():
            (debug_lump_folder / f'game_{lump_id.hex()}_old.lmp').write_bytes(game_lump.data)
            try:
                old = orig_lumps.pop(lump_id)
            except KeyError:
                LOGGER.info('{} added', lump_id)
            else:
                if old != game_lump.data:
                    LOGGER.info('{} changed', lump_id)
        if orig_lumps:
            LOGGER.info('Removed game lumps: {}', list(orig_lumps))

    try:
        from srctools.fgd import _engine_db_stats  # noqa
    except AttributeError:
        pass
    else:
        LOGGER.info('FGD database usage: {}', _engine_db_stats())

    LOGGER.info("HammerAddons postcompiler complete!")


if __name__ == '__main__':
    try:
        trio.run(main, sys.argv[1:])
    finally:
        logging_shutdown()
