"""Code for running VBSP and VRAD."""
import os.path
import sys
import subprocess
import logging

from seecompiler.logger import get_logger

LOGGER = get_logger(__name__)


def quote(txt):
    """Add quotes to text if needed."""
    if ' ' in txt:
        return '"' + txt + '"'
    return txt


def get_compiler_name(program: str):
    """Get the real executable name for VBSP or VRAD."""
    if 'win' in sys.platform:
        name = program + '_original.exe'
    elif 'darwin' in sys.platform:
        name = program + '_osx_original'
    else:
        name = program + '_linux_original'
    return quote(os.path.abspath(name))


def run_vrad(args):
    """Execute the original VRAD."""

    joined_args = get_compiler_name('vrad') + ' ' + ' '.join(map(quote, args))

    LOGGER.info("Calling original VRAD...")
    LOGGER.info('Args: {}', joined_args)

    code = subprocess.call(
        joined_args,
        shell=True,
    )
    if code == 0:
        LOGGER.info("Done!")
    else:
        LOGGER.warning("VRAD failed! (" + str(code) + ")")
        logging.shutdown()
        sys.exit(code)
