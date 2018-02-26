"""Replaces VRAD, to run operations on the final BSP."""
from zipfile import ZipFile

from seecompiler.logger import init_logging
from seecompiler.transformation import run_transformations

LOGGER = init_logging('seecompiler/vrad.log')

import sys
import os
from pkg_resources import resource_stream

from srctools.bsp import BSP, BSP_LUMPS
from srctools import GameID, FGD
from lzma import LZMAFile
from srctools.game import find_gameinfo
from seecompiler.packlist import PackList


def load_fgd() -> FGD:
    """Extract the local copy of FGD data."""
    # Pull the FGD file we included out, so we don't rely on a local one.
    with LZMAFile(resource_stream('seecompiler', 'fgd.lzma')) as f:
        return FGD.unserialise(f)


def main(argv):
    LOGGER.info('SEEcompiler VRAD hook started!')

    game_info = find_gameinfo(argv)

    fsys = game_info.get_filesystem()
    fsys.open_ref()

    packlist = PackList(fsys)

    LOGGER.info('Gameinfo: {}\nSearch path: \n{}', game_info.path, '\n'.join([sys[0].path for sys in fsys.systems]))

    fgd = load_fgd()

    LOGGER.info('Loading soundscripts...')
    packlist.load_soundscript_manifest('see_sndscript_data.vdf')
    LOGGER.info('Done! ({} sounds)', len(packlist.soundscripts))

    # The path is the last argument to VRAD
    # Hammer adds wrong slashes sometimes, so fix that.
    path = os.path.normpath(argv[-1])

    LOGGER.info("Map path is " + path)
    if path == "":
        raise Exception("No map passed!")

    if not path.endswith(".bsp"):
        path += ".bsp"

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)
    bsp_file.read_header()
    bsp_file.read_game_lumps()

    LOGGER.info('Reading entities...')
    vmf = bsp_file.read_ent_data()
    LOGGER.info('Done!')

    run_transformations(vmf, fsys)

    bsp_file.replace_lump(
        bsp_file.filename,
        BSP_LUMPS.ENTITIES,
        bsp_file.write_ent_data(vmf),
    )

    LOGGER.info('Finished writing entities.')

    packlist.pack_fgd(vmf, fgd)

    packlist.pack_from_bsp(bsp_file)
    packlist.eval_dependencies()

    with bsp_file.packfile() as pak_zip:
        packlist.pack_into_zip(pak_zip)

    LOGGER.info("SEEcompiler VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv[1:])

