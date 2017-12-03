"""Replaces VRAD, to run operations on the final BSP."""
from zipfile import ZipFile

from seecompiler.logger import init_logging
from seecompiler.transformation import run_transformations

LOGGER = init_logging('seecompiler/vrad.log')

import sys
import os
import io

from srctools.bsp import BSP, BSP_LUMPS
from srctools import FileSystemChain, GameID
from srctools.filesys import RawFileSystem, VPKFileSystem
from lzma import LZMAFile
from srctools.game import find_gameinfo
from seecompiler import packlist

import seecompiler.run


def main(argv):
    LOGGER.info('SEEcompiler VRAD hook started!')

    game_info = find_gameinfo(argv)

    fsys = game_info.get_filesystem()

    LOGGER.info('Gameinfo: {}\nSearch path: \n{}', game_info.path, '\n'.join([sys[0].path for sys in fsys.systems]))

    # The path is the last argument to VRAD
    # Hammer adds wrong slashes sometimes, so fix that.
    path = os.path.normpath(argv[-1])

    LOGGER.info("Map path is " + path)
    if path == "":
        raise Exception("No map passed!")

    if not path.endswith(".bsp"):
        path += ".bsp"

    #seecompiler.run.run_vrad(vrad_args)

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)
    bsp_file.read_header()
    bsp_file.read_game_lumps()

    LOGGER.info('Reading entities...')
    vmf = bsp_file.read_ent_data()
    LOGGER.info('Done!')

    LOGGER.info('Running transformations...')
    run_transformations(vmf, fsys, GameID.PORTAL_2)

    bsp_file.write_ent_data(vmf)
    LOGGER.info('Finished writing entities.')

    packlist.pack_from_bsp(bsp_file)
    packlist.eval_dependencies(fsys)

    pak_file = io.BytesIO(bsp_file.get_lump(BSP_LUMPS.PAKFILE))

    with ZipFile(pak_file, 'a') as pak_zip:
        seecompiler.packlist.pack_into_zip(fsys, pak_zip)

    bsp_file.replace_lump(bsp_file.filename, BSP_LUMPS.PAKFILE, pak_file.getvalue())

    LOGGER.info("SEEcompiler VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv[1:])

