import importlib
import sys
from pathlib import Path
from typing import Awaitable, Callable

import pytest

from hammeraddons.bsp_transform import Context, TransFunc, TRANSFORMS
from srctools.bsp import BSP
from srctools.filesys import FileSystemChain
from srctools.game import Game
from srctools.packlist import PackList


@pytest.fixture
def blank_ctx(shared_datadir: Path) -> Context:
    """Build a blank context."""
    bsp = BSP(shared_datadir / 'blank.bsp')
    game = Game(shared_datadir)
    fsys = FileSystemChain()
    return Context(
        fsys,
        bsp.ents,
        PackList(fsys),
        bsp,
        game,
    )


def get_transform_func(module_name: str, transform: str) -> Callable[[Context], Awaitable[None]]:
    """Import the builtin transforms, then fetch this context."""
    folder = str(Path(__file__, '..', '..', '..', 'transforms').resolve())
    print(f'Adding "{folder}" to path.')

    sys.path.append(folder)
    try:
        importlib.import_module(module_name)
        return TRANSFORMS[transform].func
    finally:
        sys.path.remove(folder)
