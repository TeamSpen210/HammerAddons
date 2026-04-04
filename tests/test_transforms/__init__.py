from collections.abc import Awaitable, Callable
from pathlib import Path
import importlib
import sys

import pytest

from hammeraddons.bsp_transform import Context, TRANSFORMS
from hammeraddons.config import GameConfig
from srctools.bsp import BSP
from srctools.filesys import FileSystemChain
from srctools.game import Game
from srctools.packlist import PackList


@pytest.fixture
def blank_ctx(shared_datadir: Path) -> Context:
    """Build a blank context."""
    bsp = BSP(shared_datadir / 'blank.bsp')
    game = Game(shared_datadir)
    game_conf = GameConfig(
        tags=frozenset(),
        steamid=None,
        io_comma_sep=False,
        instance_proxies=True,
        translucent_needs_mostlyopaque=False,
        pack_vpk=False,
        searchpaths=(),
        vscript=True,
        vscript_quote='',
        particles_manifest='',
        studiomdl_path_windows='',
        studiomdl_path_mac='',
        studiomdl_path_linux='',
    )
    fsys = FileSystemChain()
    return Context(
        fsys,
        bsp.ents,
        PackList(fsys),
        bsp,
        game,
        game_conf,
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
