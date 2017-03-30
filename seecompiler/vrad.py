"""Replaces VRAD, to run operations on the final BSP."""
from seecompiler.logger import init_logging

LOGGER = init_logging('seecompiler/vrad.log')

import sys
import os

from srctools.bsp import BSP

import seecompiler.run


def main(argv):
    LOGGER.info('SEEcompiler VRAD hook started!')
    vrad_args = argv[1:]

    # The path is the last argument to VRAD
    # Hammer adds wrong slashes sometimes, so fix that.
    path = os.path.normpath(argv[-1])

    LOGGER.info("Map path is " + path)
    if path == "":
        raise Exception("No map passed!")

    if not path.endswith(".bsp"):
        path += ".bsp"

    seecompiler.run.run_vrad(vrad_args)

    LOGGER.info('Reading BSP...')
    bsp_file = BSP(path)
    bsp_file.read_header()
    LOGGER.info('Done!')

    LOGGER.info("SEEcompiler VRAD hook finished!")

if __name__ == '__main__':
    main(sys.argv)
