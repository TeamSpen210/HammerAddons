"""Runs before VRAD, to run operations on the final BSP."""
import argparse
import datetime
import sys
from pathlib import Path

from srctools import Property
from srctools.logger import init_logging


# Put the logs in the executable folders.
LOGGER = init_logging(Path(sys.argv[0]).with_name('postcompiler.log'))

from srctools.bsp import BSP, BSP_LUMPS
from srctools.bsp_transform import run_transformations
from srctools.packlist import PackList, load_fgd
from srctools.scripts import config
from srctools.compiler import propcombine
from typing import List


def main(argv: List[str]) -> None:
    LOGGER.info('Srctools postcompiler hook started at {}!', datetime.datetime.now().isoformat())

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

    LOGGER.info("Map path is {}", path)

    conf, game_info, fsys, pack_blacklist = config.parse(path)

    fsys.open_ref()

    packlist = PackList(fsys)

    LOGGER.info('Gameinfo: {}', game_info.path)
    LOGGER.info(
        'Search paths: \n{}',
        '\n'.join([sys.path for sys, prefix in fsys.systems]),
    )

    fgd = load_fgd()

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

    bsp_file.lumps[BSP_LUMPS.ENTITIES].data = bsp_file.write_ent_data(vmf)

    if conf.get(bool, 'auto_pack') and args.allow_pack:
        LOGGER.info('Analysing packable resources...')
        packlist.pack_fgd(vmf, fgd)

        packlist.pack_from_bsp(bsp_file)

        packlist.eval_dependencies()

    packlist.pack_into_zip(bsp_file, blacklist=pack_blacklist, ignore_vpk=False)

    with bsp_file.packfile() as pak_zip:
        LOGGER.info('Packed files: \n{}'.format('\n'.join(pak_zip.namelist())))

    LOGGER.info('Writing BSP...')
    bsp_file.save()

    LOGGER.info("srctools VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv[1:])

