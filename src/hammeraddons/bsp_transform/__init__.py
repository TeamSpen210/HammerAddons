"""Transformations that can be applied to the BSP file."""
from pathlib import Path
from typing import Optional, Callable, Awaitable, Dict, Mapping, Tuple, List
import inspect

from srctools import EmptyMapping, FileSystem, Property, VMF, Output, Entity, FGD, conv_bool
from srctools.bsp import BSP
from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.game import Game


LOGGER = get_logger(__name__, 'bsp_trans')

__all__ = [
    'check_control_enabled',
    'Context', 'trans', 'run_transformations',
]


def check_control_enabled(ent: Entity) -> bool:
    """Implement the bahaviour of ControlEnables - control_type and control_value.

    This allows providing a fixup value, and optionally inverting it.
    """
    # If ctrl_type is 0, ctrl_value needs to be 1 to be enabled.
    # If ctrl_type is 1, ctrl_value needs to be 0 to be enabled.
    if 'ctrl_type' in ent:
        return conv_bool(ent['ctrl_type'], False) != conv_bool(ent['ctrl_value'], True)
    else:
        # Missing, assume true if ctrl_value also isn't present.
        return conv_bool(ent['ctrl_value'], True)


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
        self.config = Property.root()

        self._io_remaps: Dict[Tuple[str, str], Tuple[List[Output], bool]] = {}
        self._ent_code: Dict[Entity, str] = {}

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
            TRANSFORMS[name] = func  # type: ignore # inspect needs typeguard
            return func  # type: ignore # ^^^
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
    studiomdl_loc: Path=None,
    config: Mapping[str, Property]=EmptyMapping,
    fgd: FGD=None,
) -> None:
    """Run all transformations."""
    context = Context(filesys, vmf, pack, bsp, game, studiomdl_loc=studiomdl_loc, fgd=fgd)

    for func_name, func in sorted(
        TRANSFORMS.items(),
        key=lambda tup: TRANSFORM_PRIORITY[tup[0]],
    ):
        LOGGER.info('Running "{}"...', func_name)
        try:
            context.config = config[func_name.casefold()]
        except KeyError:
            context.config = Property(func_name, [])
        LOGGER.debug('Config: {!r}', context.config)
        await func(context)

    if context._ent_code:
        LOGGER.info('Injecting VScript code...')
        for ent, code in context._ent_code.items():
            init_scripts = ent['vscripts'].split()
            init_scripts.append(pack.inject_vscript(code.replace('`', '"')))
            ent['vscripts'] = ' '.join(init_scripts)

    if context._io_remaps:
        LOGGER.info('Remapping outputs...')
        for (name, inp_name), outs in context._io_remaps.items():
            LOGGER.debug('Remap {}.{} = {}', name, inp_name, outs)
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
                            times=out.times if rep_out.times == -1
                            else rep_out.times if out.times == -1
                            else min(out.times, rep_out.times),
                        )
                        ent.outputs.append(new_out)
                        deferred.append(new_out)
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
    from . import (
        globals,
        instancing,
        packing,
    )


_load()
