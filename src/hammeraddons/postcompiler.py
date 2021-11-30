"""Runs before VRAD, to run operations on the final BSP."""
import argparse
import os
import sys
import warnings
from collections import defaultdict
from logging import FileHandler
from pathlib import Path
from typing import List, Dict, Optional

from srctools.logger import init_logging, Formatter


# Put the logs in the executable folders.
LOGGER = init_logging(Path(sys.argv[0]).with_name('postcompiler.log'))
warnings.filterwarnings(category=DeprecationWarning, module='srctools', action='once')

from srctools import Property, __version__ as version_lib
from srctools.filesys import ZipFileSystem
from srctools.fgd import FGD
from srctools.bsp import BSP
from srctools.bsp_transform import run_transformations
from srctools.packlist import PackList
from srctools.scripts import config
from srctools.compiler import propcombine, __version__ as version_haddons


def main(argv: List[str]) -> None:
    """Run the postcompiler."""
    parser = argparse.ArgumentParser(
        description="Modifies the BSP file, allowing additional entities "
                    "and bugfixes.",
    )

    parser.add_argument(
        "-game", "--game",
        dest="game_folder",
        default="",
        help="Specify the folder containing gameinfo.txt, and thus the "
             "location of the game. This overrides the option specified "
             "in srctools.vdf.",
    )

    parser.add_argument(
        "--nopack",
        dest="allow_pack",
        action="store_false",
        help="Prevent packing of files found in the map."
    )
    parser.add_argument(
        "--propcombine",
        action="store_true",
        help="Allow merging static props together.",
    )
    parser.add_argument(
        "--showgroups",
        action="store_true",
        help="Show propcombined props, by setting their tint to random groups",
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

    # The path is the last argument to the compiler.
    # Hammer adds wrong slashes sometimes, so fix that.
    # Also if it's the VMF file, make it the BSP.
    path = Path(args.map).with_suffix('.bsp')

    # Open and start writing to the map's log file.
    handler = FileHandler(path.with_suffix('.log'))
    handler.setFormatter(Formatter(
        # One letter for level name
        '[{levelname}] {module}.{funcName}(): {message}',
        style='{',
    ))
    LOGGER.addHandler(handler)

    LOGGER.info('HammerAddons postcompiler, srctools=v{}, addons=v{}', version_lib, version_haddons)
    LOGGER.info("Map path is {}", path)

    (
        conf, game_info,
        fsys, pack_blacklist,
        plugin,
    ) = config.parse(path, args.game_folder)

    LOGGER.info('Loading plugins...')
    plugin.load_all()

    packlist = PackList(fsys)

    LOGGER.info('Gameinfo: {}', game_info.path)
    LOGGER.info(
        'Search paths: \n{}',
        '\n'.join([system.path for system, prefix in fsys.systems]),
    )

    fgd = FGD.engine_dbase()

    LOGGER.info('Loading soundscripts...')
    packlist.load_soundscript_manifest(
        conf.path.with_name('srctools_sndscript_data.vdf')
    )
    LOGGER.info('Done! ({} sounds)', len(packlist.soundscript))
    LOGGER.info('Loading particles...')
    packlist.load_particle_manifest()
    LOGGER.info('Done! ({} particles)', len(packlist.particles))
    LOGGER.debug('Known particles: \n{}', "\n".join([
        f'{fname}: {mode.value}' for fname, mode in
        packlist.particles._files.items()
    ]))

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)

    LOGGER.info('Reading entities...')
    LOGGER.info('Done!')

    # Mount the existing packfile, so the cubemap files are recognised.
    LOGGER.info('Mounting BSP packfile...')
    fsys.add_sys(ZipFileSystem('<BSP pakfile>', bsp_file.pakfile))

    studiomdl_path = conf.get(str, 'studiomdl')
    studiomdl_loc: Optional[Path]
    if studiomdl_path:
        studiomdl_loc = (game_info.root / studiomdl_path).resolve()
        if not studiomdl_loc.exists():
            LOGGER.warning('No studiomdl found at "{}"!', studiomdl_loc)
            studiomdl_loc = None
    else:
        LOGGER.warning('No studiomdl path provided.')
        studiomdl_loc = None

    use_comma_sep = conf.get(bool, 'use_comma_sep')
    if use_comma_sep is None:
        # Guess the format, by checking existing outputs.
        used_comma_sep = {
            out.comma_sep
            for ent in bsp_file.ents.entities
            for out in ent.outputs
        }
        try:
            [use_comma_sep] = used_comma_sep
        except ValueError:
            if used_comma_sep:
                LOGGER.warning("Both BSP I/O formats in map? This shouldn't be possible.")
            else:
                LOGGER.warning('No outputs in map, could not determine BSP I/O format!')
            LOGGER.warning('Set "use_comma_sep" in srctools.vdf.')
            use_comma_sep = False  # Kinda arbitrary.
    transform_conf = {
        prop.name: prop
        for prop in conf.get(Property, 'transform_opts')
    }

    LOGGER.info('Running transforms...')
    run_transformations(
        bsp_file.ents,
        fsys, packlist,
        bsp_file,
        game_info,
        studiomdl_loc,
        transform_conf,
    )

    if studiomdl_loc is not None and args.propcombine:
        decomp_cache_path = conf.get(str, 'propcombine_cache')
        decomp_cache_loc: Optional[Path]
        crowbar_loc: Optional[Path]
        if decomp_cache_path is not None:
            decomp_cache_loc = (game_info.root / decomp_cache_path).resolve()
            decomp_cache_loc.mkdir(parents=True, exist_ok=True)
        else:
            decomp_cache_loc = None
        if conf.get(bool, 'propcombine_crowbar'):
            # argv[0] is the location of our script/exe, which lets us locate
            # Crowbar from there. The environment var is for testing.
            if 'CROWBAR_LOC' in os.environ:
                crowbar_loc = Path(os.environ['CROWBAR_LOC']).resolve()
            else:
                crowbar_loc = Path(sys.argv[0], '../Crowbar.exe').resolve()
        else:
            crowbar_loc = None

        LOGGER.info('Combining props...')
        propcombine.combine(
            bsp_file,
            bsp_file.ents,
            packlist,
            game_info,
            studiomdl_loc,
            qc_folders=[
                game_info.root / folder
                for folder in
                conf.get(Property, 'propcombine_qc_folder').as_array(conv=Path)
            ],
            decomp_cache_loc=decomp_cache_loc,
            crowbar_loc=crowbar_loc,
            auto_range=conf.get(int, 'propcombine_auto_range'),
            min_cluster=conf.get(int, 'propcombine_min_cluster'),
            blacklist=conf.get(Property, 'propcombine_blacklist').as_array(),
            volume_tolerance=conf.get(float, 'propcombine_volume_tolerance'),
            debug_tint=args.showgroups,
            debug_dump=args.dumpgroups,
        )
        LOGGER.info('Done!')
    else:  # Strip these if they're present.
        for ent in bsp_file.ents.by_class['comp_propcombine_set']:
            ent.remove()
        for ent in bsp_file.ents.by_class['comp_propcombine_volume']:
            bsp_file.bmodels.pop(ent, None)  # Ignore if not present.
            ent.remove()

    if conf.get(bool, 'auto_pack') and args.allow_pack:
        LOGGER.info('Analysing packable resources...')
        packlist.pack_fgd(bsp_file.ents, fgd)

        packlist.pack_from_bsp(bsp_file)

        packlist.eval_dependencies()
        if conf.get(bool, 'soundscript_manifest'):
            packlist.write_soundscript_manifest()
        man_name = conf.get(str, 'particles_manifest')
        if man_name:
            man_name = man_name.replace('<map name>', path.stem)
            LOGGER.info('Writing particle manifest "{}"...', man_name)
            packlist.write_particles_manifest(man_name)

    dump_path = conf.get(str, 'pack_dump')
    if dump_path:
        packlist.pack_into_zip(
            bsp_file,
            blacklist=pack_blacklist,
            ignore_vpk=False,
            dump_loc=Path(game_info.root, dump_path.lstrip('#')).absolute().resolve(),
            only_dump=dump_path.startswith('#'),
        )
    else:
        packlist.pack_into_zip(
            bsp_file,
            blacklist=pack_blacklist,
            ignore_vpk=False,
        )

    # List out all the files, but group together files with the same extension.
    ext_for_name: Dict[str, List[str]] = defaultdict(list)
    for file in bsp_file.pakfile.infolist():
        filename = Path(file.filename)
        if '.' in filename.name:
            stem, ext = filename.name.split('.', 1)
            file_path = str(filename.parent / stem)
        else:
            file_path = file.filename
            ext = ''

        ext_for_name[file_path].append(ext)

    LOGGER.info('Packed files: \n{}'.format('\n'.join([
        (
            f'{name}.{exts[0]}'
            if len(exts) == 1 else
            f'{name}.({"/".join(exts)})')
        for name, exts in sorted(ext_for_name.items())
    ])))

    LOGGER.info('Writing BSP...')
    bsp_file.save()

    LOGGER.info("HammerAddons postcompiler complete!")

if __name__ == '__main__':
    main(sys.argv[1:])
