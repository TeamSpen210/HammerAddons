"""Transformations that can be applied to the BSP file."""
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Any

from srctools import FileSystem, VMF, Output, Entity, FGD
from srctools.bsp import BSP
from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.game import Game


LOGGER = get_logger(__name__, 'bsp_trans')

__all__ = ['Context', 'trans', 'run_transformations']


class Context:
    """Bundles information useful for each transformation.

    This allows them to ignore data they don't use.
    """
    def __init__(
        self, 
        filesys: FileSystem,
        vmf: VMF,
        pack: PackList,
        bsp: BSP,
        game: Game,
        *,
        fgd: FGD = None,
        studiomdl_loc: Path=None,
    ) -> None:
        self.sys = filesys
        self.vmf = vmf
        self.bsp = bsp
        self.pack = pack
        self.bsp_path = Path(bsp.filename)
        self.fgd = fgd or FGD.engine_dbase()
        self.game = game
        self.studiomdl = studiomdl_loc

        self._io_remaps = {}  # type: Dict[Tuple[str, str], Tuple[List[Output], bool]]
        self._ent_code = {}  # type: Dict[Entity, str]

    def add_io_remap(self, name: str, *outputs: Output, remove: bool=True) -> None:
        """Register an output to be replaced.

        This is used to convert inputs to comp_ entities into their real
        forms. The output name in the output is the input that will be replaced.

        If remove is set to False, the original output will be kept.
        If the name is blank, this does nothing.
        """
        if not name:
            return

        name = name.casefold()
        for out in outputs:
            inp_name = out.output.casefold()
            out.output = ''
            key = (name, inp_name)
            try:
                out_list, old_remove = self._io_remaps[key]
            except KeyError:
                self._io_remaps[key] = ([out], remove)
            else:
                out_list.append(out)
                # Only allow removing if all remaps have requested it.
                if old_remove and not remove:
                    self._io_remaps[key] = (out_list, False)

    def add_io_remap_removal(self, name: str, inp_name: str) -> None:
        """Special case of add_io_remap, request that this output should be removed."""
        key = (name.casefold(), inp_name.casefold())
        if key not in self._io_remaps:
            self._io_remaps[key] = ([], True)

    def add_code(self, ent: Entity, code: str) -> None:
        """Register VScript code to be run on spawn for this entity.

        This way multiple such options can be merged together.
        """
        try:
            existing = self._ent_code[ent]
        except KeyError:
            self._ent_code[ent] = code
        else:
            self._ent_code[ent] = '{}\n{}'.format(existing, code)


TransFunc = Callable[[Context], None]
TRANSFORMS = {}  # type: Dict[str, TransFunc]


def trans(name: str, *, priority: int=0) -> Callable[[TransFunc], TransFunc]:
    """Add a transformation procedure to the list."""
    def deco(func: TransFunc) -> TransFunc:
        """Stores the transformation."""
        TRANSFORMS[name] = func
        func.priority = priority
        return func
    return deco


# noinspection PyProtectedMember
def run_transformations(
    vmf: VMF,
    filesys: FileSystem,
    pack: PackList,
    bsp: BSP,
    game: Game,
    studiomdl_loc: Path=None,
) -> None:
    """Run all transformations."""
    context = Context(filesys, vmf, pack, bsp, game, studiomdl_loc=studiomdl_loc)

    for func_name, func in sorted(
        TRANSFORMS.items(),
        key=lambda tup: tup[1].priority,
    ):
        LOGGER.info('Running "{}"...', func_name)
        func(context)

    if context._ent_code:
        LOGGER.info('Injecting VScript code...')
        for ent, code in context._ent_code.items():
            init_scripts = ent['vscripts'].split()
            init_scripts.append(pack.inject_vscript(code.replace('`', '"')))
            ent['vscripts'] = ' '.join(init_scripts)

    if context._io_remaps:
        LOGGER.info('Remapping outputs...')
        for ent in vmf.entities:
            todo = ent.outputs[:]
            # Recursively convert only up to 500 times.
            # Arbitrary limit, should be sufficient.
            for _ in range(500):
                if not todo:
                    break
                deferred = []
                for out in todo:
                    try:
                        remaps, should_remove = context._io_remaps[
                            out.target.casefold(),
                            out.input.casefold(),
                        ]
                    except KeyError:
                        continue
                    if should_remove:
                        ent.outputs.remove(out)
                    for rep_out in remaps:
                        new_out = Output(
                            out.output,
                            rep_out.target,
                            rep_out.input,
                            rep_out.params or out.params,
                            out.delay + rep_out.delay,
                            only_once=rep_out.only_once and out.only_once,
                        )
                        ent.outputs.append(new_out)
                        deferred.append(new_out)
                todo = deferred
            else:
                LOGGER.error(
                    'Entity "{}" ({}) has infinite loop when expanding '
                    ' compiler outputs to real ones! Final output list: \n{}',
                    ent['targetname'], ent['classname'],
                    '\n'.join(['* {}\n'.format(out) for out in ent.outputs])
                )


def _load() -> None:
    """Import all submodules.

    This loads the transformations. We do it in a function to allow discarding
    the output.
    """
    from srctools.bsp_transform import (
        globals,
        instancing,
        packing,
        tweaks,
    )
_load()
