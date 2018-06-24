"""Transformations that can be applied to the BSP file."""
from typing import Callable, TypeVar

from srctools import FileSystem, VMF
from srctools.logger import get_logger

LOGGER = get_logger(__name__, 'bsp_trans')

__all__ = ['Context', 'trans', 'run_transformations']

TRANSFORMS = {}


class Context:
    """Bundles information useful for each transformation.

    This allows them to ignore data they don't use.
    """
    def __init__(
        self, 
        filesys: FileSystem,
        vmf: VMF,
    ):
        self.sys = filesys
        self.vmf = vmf


TransFunc = Callable[[Context], None]


def trans(name: str) -> Callable[[TransFunc], TransFunc]:
    """Add a transformation procedure to the list."""
    def deco(func: TransFunc) -> TransFunc:
        """Stores the transformation."""
        TRANSFORMS[name] = func
        return func
    return deco


def run_transformations(vmf: VMF, filesys: FileSystem) -> None:
    """Run all transformations."""
    context = Context(
        filesys,
        vmf,
    )
    for func_name, func in TRANSFORMS.items():
        LOGGER.info('Running "{}"...', func_name)
        func(context)

# Import the modules.
from srctools.bsp_transform import (
    antline,
    brush_ents,
    globals,
    portal2,
    sceneset,
    instancing,
)
