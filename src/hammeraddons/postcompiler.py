"""Runs before VRAD, to run operations on the final BSP."""
import argparse
import datetime
import sys
from collections import defaultdict
from logging import FileHandler
from pathlib import Path

from srctools import Property
from srctools.logger import init_logging, Formatter


# Put the logs in the executable folders.
LOGGER = init_logging(Path(sys.argv[0]).with_name('postcompiler.log'))

from srctools.fgd import FGD
from srctools.bsp import BSP, BSP_LUMPS
from srctools.bsp_transform import run_transformations
from srctools.packlist import PackList
from srctools.scripts import config
from srctools.compiler import propcombine
from typing import List, Dict


def main(argv: List[str]) -> None:

    parser = argparse.ArgumentParser(
        description="Modifies the BSP file, allowing additional entities "
                    "and bugfixes.",
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
        help="Show propcombined props, by setting their tint to 0 255 0",
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

    LOGGER.info('Srctools postcompiler hook started at {}!', datetime.datetime.now().isoformat())
    LOGGER.info("Map path is {}", path)

    conf, game_info, fsys, pack_blacklist, plugins = config.parse(path)

    fsys.open_ref()

    packlist = PackList(fsys)

    LOGGER.info('Gameinfo: {}', game_info.path)
    LOGGER.info(
        'Search paths: \n{}',
        '\n'.join([sys.path for sys, prefix in fsys.systems]),
    )

    fgd = FGD.engine_dbase()

    LOGGER.info('Loading soundscripts...')
    packlist.load_soundscript_manifest(
        conf.path.with_name('srctools_sndscript_data.vdf')
    )
    LOGGER.info('Done! ({} sounds)', len(packlist.soundscripts))

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)

    LOGGER.info('Reading entities...')
    vmf = bsp_file.read_ent_data()
    LOGGER.info('Done!')

    studiomdl_path = conf.get(str, 'studiomdl')
    if studiomdl_path:
        studiomdl_loc = (game_info.root / studiomdl_path).resolve()
        if not studiomdl_loc.exists():
            LOGGER.warning('No studiomdl found at "{}"!', studiomdl_loc)
            studiomdl_loc = None
    else:
        LOGGER.warning('No studiomdl path provided.')
        studiomdl_loc = None

    for plugin in plugins:
        plugin.load()

    use_comma_sep = conf.get(bool, 'use_comma_sep')
    if use_comma_sep is None:
        # Guess the format, by picking whatever the first output uses.
        for ent in vmf.entities:
            for out in ent.outputs:
                use_comma_sep = out.comma_sep
                break
        if use_comma_sep is None:
            LOGGER.warning('No outputs in map, could not determine BSP I/O format!')
            LOGGER.warning('Set "use_comma_sep" in srctools.vdf.')
        use_comma_sep = False

    run_transformations(vmf, fsys, packlist, bsp_file, game_info, studiomdl_loc)

    if studiomdl_loc is not None and args.propcombine:
        LOGGER.info('Combining props...')
        propcombine.combine(
            bsp_file,
            vmf,
            packlist,
            game_info,
            studiomdl_loc,
            [
                game_info.root / folder
                for folder in
                conf.get(Property, 'propcombine_qc_folder').as_array(conv=Path)
            ],
            conf.get(int, 'propcombine_auto_range'),
            conf.get(int, 'propcombine_min_cluster'),
            debug_tint=args.showgroups,
        )
        LOGGER.info('Done!')
    else:  # Strip these if they're present.
        for ent in vmf.by_class['comp_propcombine_set']:
             ent.remove()

    bsp_file.lumps[BSP_LUMPS.ENTITIES].data = bsp_file.write_ent_data(vmf, use_comma_sep)

    if conf.get(bool, 'auto_pack') and args.allow_pack:
        LOGGER.info('Analysing packable resources...')
        packlist.pack_fgd(vmf, fgd)

        packlist.pack_from_bsp(bsp_file)

        packlist.eval_dependencies()

    packlist.pack_into_zip(bsp_file, blacklist=pack_blacklist, ignore_vpk=False)

    with bsp_file.packfile() as pak_zip:
        # List out all the files, but group together files with the same extension.
        ext_for_name: Dict[str, List[str]] = defaultdict(list)
        for file in pak_zip.infolist():
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

    LOGGER.info("srctools VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv[1:])

