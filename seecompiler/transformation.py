"""Infrastructure for calling all the different transformations to the BSP."""
from srctools import GameID, FileSystem, VMF

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
    for func in TRANSFORMS.values():
        trans_game = getattr(func, 'game_id', None)
        if trans_game is not None and trans_game is not game_id:
            continue
        func(context)

# Import the modules.
from seecompiler import trans_brush_ents
