"""Transformations that can be applied to the BSP file."""
from typing import Protocol
from collections.abc import Awaitable, Callable, Container, Mapping
from pathlib import Path
import inspect
import warnings

import attrs
import trio.lowlevel

from srctools import FGD, VMF, EmptyMapping, Entity, FileSystem, Keyvalues, Output
from srctools.bsp import BSP
from srctools.game import Game
from srctools.logger import get_logger
from srctools.packlist import PackList

from hammeraddons.bsp_transform.common import (
    check_control_enabled, ent_description,
    parse_numeric_specifier, NumericSpecifier, NumericOp
)

LOGGER = get_logger(__name__, 'bsp_trans')
type RemapFunc = Callable[[Entity, Output], list[Output]]

__all__ = [
    'Context', 'trans', 'run_transformations',
    'TransFunc', 'TRANSFORMS',
    # Utils:
    'check_control_enabled', 'ent_description',
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
        studiomdl_loc: Path | None = None,
        tags: frozenset[str] = frozenset(),
        modelcompile_dump: Path | None = None,
    ) -> None:
        self.sys = filesys
        self.vmf = vmf
        self.bsp = bsp
        self.pack = pack
        self.bsp_path = Path(bsp.filename)
        self._fgd: FGD | None = None
        self.tags = tags
        self.modelcompile_dump = modelcompile_dump
        self.game = game
        self.studiomdl = studiomdl_loc
        self.config = Keyvalues.root()

        self._io_remaps: dict[tuple[str, str], tuple[list[Output | RemapFunc], bool]] = {}
        self._allow_remaps = True
        self._ent_code: dict[Entity, str] = {}

    @property
    def fgd(self) -> FGD:
        warnings.warn("Use EntityDef.engine_def() if possible.")
        if self._fgd is None:
            self._fgd = FGD.engine_dbase()
        return self._fgd

    def _add_io_remap(
        self, name: str, inp_name: str,
        value: Output | RemapFunc,
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
            self._ent_code[ent] = f'{existing}\n{code}'


type TransFunc = Callable[[Context], Awaitable[None]]
type TransFuncOrSync = Callable[[Context], Awaitable[None] | None]


@attrs.frozen(eq=False)
class Transform:
    """A transform function."""
    func: TransFunc
    name: str
    priority: int


TRANSFORMS: dict[str, Transform] = {}


class TransProto(Protocol):
    def __call__[Func: TransFuncOrSync](self, func: Func) -> Func: ...


def trans(name: str, *, priority: int=0) -> TransProto:
    """Add a transformation procedure to the list."""
    name = name.strip()
    if ',' in name:
        raise ValueError('Commas are not allowed in names!')

    def deco[Func: TransFuncOrSync](func: Func) -> Func:
        """Stores the transformation."""
        if inspect.iscoroutinefunction(func):
            TRANSFORMS[name.casefold()] = Transform(func, name, priority)
        else:
            async def async_wrapper(ctx: Context) -> None:
                """Just freeze all other tasks to run this."""
                await trio.lowlevel.checkpoint()
                func(ctx)
            TRANSFORMS[name.casefold()] = Transform(async_wrapper, name, priority)
        return func
    return deco


# noinspection PyProtectedMember
async def run_transformations(
    vmf: VMF,
    filesys: FileSystem,
    pack: PackList,
    bsp: BSP,
    game: Game,
    studiomdl_loc: Path | None = None,
    config: Mapping[str, Keyvalues] = EmptyMapping,
    tags: frozenset[str] = frozenset(),
    disabled: Container[str] = (),
    modelcompile_dump: Path | None = None,
) -> None:
    """Run all transformations."""
    context = Context(
        filesys, vmf, pack, bsp, game,
        studiomdl_loc=studiomdl_loc, tags=tags,
        modelcompile_dump=modelcompile_dump,
    )

    for transform in sorted(TRANSFORMS.values(), key=lambda trans: trans.priority):
        if transform.name.casefold() in disabled:
            LOGGER.info('Skipping "{}"', transform.name)
            continue
        LOGGER.info('Running "{}"...', transform.name)
        try:
            context.config = config[transform.name.casefold()]
        except KeyError:
            context.config = Keyvalues(transform.name, [])
        LOGGER.debug('Config: {!r}', context.config)
        await transform.func(context)

    if context._ent_code:
        LOGGER.info('Injecting VScript code...')
        for ent, code in context._ent_code.items():
            init_scripts = ent['vscripts'].split()
            if init_scripts:
                # If both a regular entity script and injected script are present,
                # the call chaining mechanism used for Precache & OnPostSpawn can malfunction.
                # If the entity script defines either, running the second script will make the
                # chainer pick it up again, calling the function twice. So edit the script to
                # first blank out the functions.
                code = 'OnPostSpawn<-Precache<-function(){}\n' + code
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
                collapsed_remaps: list[Output] = []
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
                '\n'.join([f'* {out}\n' for out in ent.outputs])
            )


def _load() -> None:
    """Import all submodules.

    This loads the transformations. We do it in a function to allow discarding
    the output.
    """
    from . import globals, instancing, packing  # noqa


_load()
