"""Infrastructure for calling all the different transformations to the BSP."""
import pkgutil
import importlib

from srctools import GameID, FileSystem, VMF
from seecompiler.logger import get_logger
import seecompiler

LOGGER = get_logger(__name__)

# The things 'user' modules want to have access to easily. 
__all__ = ['Context', 'trans', 'GameID']

TRANSFORMS = {}


class Context:
    def __init__(
        self, 
        filesys: FileSystem,
        vmf: VMF,
        game_id: GameID,
    ):
        self.game = game_id
        self.sys = filesys
        self.vmf = vmf


def trans(name, game_id=None):
    """Add a transformation procedure to the list.
    
    If game_id is set, the game must be the given one to apply.
    """
    def deco(func):
        TRANSFORMS[name] = func
        func.game_id = game_id
        return func
    return deco


def run_transformations(vmf: VMF, filesys: FileSystem, game_id: GameID):
    context = Context(
        filesys,
        vmf,
        game_id,
    )
    for func_name, func in TRANSFORMS.items():
        trans_game = getattr(func, 'game_id', None)
        if trans_game is not None and trans_game is not game_id:
            continue

        LOGGER.info('Running "{}"...', func_name)

        func(context)

# Import the modules.
# noinspection PyUnresolvedReferences
from seecompiler import (
    trans_brush_ents,
    trans_antline,
    trans_globals,
    trans_sceneset,
    trans_p2,
)

