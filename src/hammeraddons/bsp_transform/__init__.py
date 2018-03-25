"""Transformations that can be applied to the BSP file."""
from srctools import FileSystem, VMF
from seecompiler.logger import get_logger

LOGGER = get_logger(__name__)

# The things 'user' modules want to have access to easily. 
__all__ = ['Context', 'trans']

TRANSFORMS = {}

class Context:
    def __init__(
        self, 
        filesys: FileSystem,
        vmf: VMF,
    ):
        self.sys = filesys
        self.vmf = vmf


def trans(name):
    """Add a transformation procedure to the list."""
    def deco(func):
        TRANSFORMS[name] = func
        return func
    return deco


def run_transformations(vmf: VMF, filesys: FileSystem):
    context = Context(
        filesys,
        vmf,
    )
    for func_name, func in TRANSFORMS.items():
        LOGGER.info('Running "{}"...', func_name)
        func(context)

# Import the modules.
# noinspection PyUnresolvedReferences
from srctools.bsp_transform import (
    antline,
    brush_ents,
    globals,
    portal2,
    sceneset,
)
