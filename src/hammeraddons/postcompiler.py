"""Runs before VRAD, to run operations on the final BSP."""
from srctools.logger import init_logging
from pathlib import Path
import sys

# Put the logs in the executable folders.
LOGGER = init_logging(Path(sys.argv[0]).with_name('postcompiler.log'))

from srctools.bsp import BSP, BSP_LUMPS
from srctools.bsp_transform import run_transformations
from srctools.packlist import PackList, load_fgd
from srctools.scripts import config
from srctools.compiler import propcombine
from typing import List


def main(argv: List[str]) -> None:
    LOGGER.info('Srctools postcompiler hook started!')

    if len(argv) == 0:
        raise Exception("No map passed!")

    # The path is the last argument to the compiler.
    # Hammer adds wrong slashes sometimes, so fix that.
    # Also if it's the VMF file, make it the BSP.
    path = Path(argv[-1]).with_suffix('.bsp')

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

    run_transformations(vmf, fsys, packlist)

    studiomdl_loc = conf.get(str, 'propcombine_studiomdl')
    if studiomdl_loc:
        LOGGER.info('Combining props...')
        propcombine.combine(
            bsp_file,
            packlist,
            game_info,
            game_info.root / studiomdl_loc,
            game_info.root / conf.get(str, 'propcombine_qc_folder'),
        )
        LOGGER.info('Done!')

    bsp_file.lumps[BSP_LUMPS.ENTITIES].data = bsp_file.write_ent_data(vmf)

    if conf.get(bool, 'auto_pack'):
        LOGGER.info('Analysing packable resources...')
        packlist.pack_fgd(vmf, fgd)

        packlist.pack_from_bsp(bsp_file)

        packlist.eval_dependencies()

    with bsp_file.packfile() as pak_zip:
        packlist.pack_into_zip(pak_zip, blacklist=pack_blacklist)

    LOGGER.info('Packed files: \n{}'.format('\n'.join(pak_zip.namelist())))

    LOGGER.info('Writing BSP...')
    bsp_file.save()

    LOGGER.info("srctools VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv[1:])

