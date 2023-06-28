"""Transformations that can be applied to the BSP file."""
from typing import Awaitable, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union
from typing_extensions import TypeAlias
import warnings
from pathlib import Path
import inspect


from srctools import FGD, VMF, EmptyMapping, Entity, FileSystem, Keyvalues, Output
from srctools.bsp import BSP
from srctools.game import Game
from srctools.logger import get_logger
from srctools.packlist import PackList

from hammeraddons.bsp_transform.common import (
    check_control_enabled,
    parse_numeric_specifier, NumericSpecifier, NumericOp
)

LOGGER = get_logger(__name__, 'bsp_trans')
RemapFunc: TypeAlias = Callable[[Entity, Output], List[Output]]

__all__ = [
    'Context', 'trans', 'run_transformations',
    'TransFunc', 'TRANSFORMS',
    # Utils:
    'check_control_enabled',
    'parse_numeric_specifier', 'NumericOp', 'NumericSpecifier',
]


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
        studiomdl_loc: Optional[Path] = None,
        tags: FrozenSet[str] = frozenset(),
        modelcompile_dump: Optional[Path] = None,
    ) -> None:
        self.sys = filesys
        self.vmf = vmf
        self.bsp = bsp
        self.pack = pack
        self.bsp_path = Path(bsp.filename)
        self._fgd: Optional[FGD] = None
        self.tags = tags
        self.modelcompile_dump = modelcompile_dump
        self.game = game
        self.studiomdl = studiomdl_loc
        self.config = Keyvalues.root()

        self._io_remaps: Dict[Tuple[str, str], Tuple[List[Union[Output, RemapFunc]], bool]] = {}
        self._allow_remaps = True
        self._ent_code: Dict[Entity, str] = {}

    @property
    def fgd(self) -> FGD:
        warnings.warn("Use EntityDef.engine_def() if possible.")
        if self._fgd is None:
            self._fgd = FGD.engine_dbase()
        return self._fgd

    def _add_io_remap(
        self, name: str, inp_name: str,
        value: Union[Output, RemapFunc],
        remove: bool,
    ) -> None:
        if not self._allow_remaps:
            raise RecursionError('Cannot add more remaps from a remap callback!')
        key = (name, inp_name)
        try:
            out_list, old_remove = self._io_remaps[key]
        except KeyError:
            self._io_remaps[key] = ([value], remove)
        else:
            out_list.append(value)
            # Only allow removing if all remaps have requested it.
            if old_remove and not remove:
                self._io_remaps[key] = (out_list, False)

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
            self._add_io_remap(name, inp_name, out, remove)

    def add_io_remap_func(self, name: str, inp_name: str, func: RemapFunc, remove: bool = True) -> None:
        """Register an output to be dynamically replaced, using a function.

        This allows varying the output for each input entity. The entity and the relevant output
        are passed to the function for reference, but the output and entity should not be modified.
        Instead, return new outputs from the function, which are merged with the original.
        """
        if name and inp_name:
            self._add_io_remap(name.casefold(), inp_name.casefold(), func, remove)

    def add_io_remap_removal(self, name: str, inp_name: str) -> None:
        """Special case of add_io_remap, request that this output should be removed."""
        if not self._allow_remaps:
            raise RecursionError('Cannot add more remaps from a remap callback!')

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


TransFunc = Callable[[Context], Awaitable[None]]
TransFuncOrSync = Callable[[Context], Optional[Awaitable[None]]]
TRANSFORMS: Dict[str, TransFunc] = {}
TRANSFORM_PRIORITY: Dict[str, int] = {}


def trans(name: str, *, priority: int=0) -> Callable[[TransFuncOrSync], TransFunc]:
    """Add a transformation procedure to the list."""
    def deco(func: TransFuncOrSync) -> TransFunc:
        """Stores the transformation."""
        TRANSFORM_PRIORITY[name] = priority
        if inspect.iscoroutinefunction(func):
            TRANSFORMS[name] = func
            return func
        else:
            async def async_wrapper(ctx: Context) -> None:
                """Just freeze all other tasks to run this."""
                func(ctx)
            TRANSFORMS[name] = async_wrapper
            return async_wrapper
    return deco


# noinspection PyProtectedMember
async def run_transformations(
    vmf: VMF,
    filesys: FileSystem,
    pack: PackList,
    bsp: BSP,
    game: Game,
    studiomdl_loc: Optional[Path] = None,
    config: Mapping[str, Keyvalues] = EmptyMapping,
    tags: FrozenSet[str] = frozenset(),
    modelcompile_dump: Optional[Path] = None,
) -> None:
    """Run all transformations."""
    context = Context(
        filesys, vmf, pack, bsp, game,
        studiomdl_loc=studiomdl_loc, tags=tags,
        modelcompile_dump=modelcompile_dump,
    )

    for func_name, func in sorted(
        TRANSFORMS.items(),
        key=lambda tup: TRANSFORM_PRIORITY[tup[0]],
    ):
        LOGGER.info('Running "{}"...', func_name)
        try:
            context.config = config[func_name.casefold()]
        except KeyError:
            context.config = Keyvalues(func_name, [])
        LOGGER.debug('Config: {!r}', context.config)
        await func(context)

    if context._ent_code:
        LOGGER.info('Injecting VScript code...')
        for ent, code in context._ent_code.items():
            init_scripts = ent['vscripts'].split()
            init_scripts.append(pack.inject_vscript(code.replace('`', '"')))
            ent['vscripts'] = ' '.join(init_scripts)

    apply_io_remaps(context)


# noinspection PyProtectedMember
def apply_io_remaps(context: Context) -> None:
    """Apply all the IO remaps."""
    # Always disallow remaps now.
    context._allow_remaps = False

    if not context._io_remaps:
        return

    LOGGER.info('Remapping outputs...')
    for (name, inp_name), outs in context._io_remaps.items():
        LOGGER.debug('Remap {}.{} = {}', name, inp_name, outs)

    for ent in context.vmf.entities:
        if not ent.outputs:  # Early out.
            continue
        todo = ent.outputs[:]
        # Recursively convert only up to 500 times.
        # Arbitrary limit, should be sufficient.
        for _ in range(500):
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
                collapsed_remaps: List[Output] = []
                out_copy = out.copy()  # Don't allow remapping functions to modify this.
                for remap in remaps:
                    if isinstance(remap, Output):
                        collapsed_remaps.append(remap)
                    else:
                        collapsed_remaps.extend(remap(ent, out_copy))

                for rep_out in collapsed_remaps:
                    new_out = Output.combine(out, rep_out)
                    ent.outputs.append(new_out)
                    deferred.append(new_out)
            if not deferred:
                break
            todo = deferred
        else:
            LOGGER.error(
                'Entity "{}" ({}) @ {} has infinite loop when expanding '
                ' compiler outputs to real ones! Final output list: \n{}',
                ent['targetname'], ent['classname'], ent['origin'],
                '\n'.join(['* {}\n'.format(out) for out in ent.outputs])
            )


def _load() -> None:
    """Import all submodules.

    This loads the transformations. We do it in a function to allow discarding
    the output.
    """
    from . import globals, instancing, packing


_load()
