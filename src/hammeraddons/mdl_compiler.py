"""Manages potential models that are being generated.

Each comes with a key, used to identify a previously compiled version.
We can then reuse already compiled versions.
"""
from typing import Self
from collections.abc import Awaitable, Callable, Hashable
from pathlib import Path
import shutil
import contextlib
import os
import pickle
import random
import tempfile
import types

from srctools import AtomicWriter, logger
from srctools.game import Game
from srctools.mdl import MDL_EXTS
from srctools.packlist import PackList
import trio

from hammeraddons.acache import ACache
from hammeraddons.bsp_transform import Context


LOGGER = logger.get_logger(__name__)
force_regen = False  # If set, force every model to be regenerated.


class GenModel[OutT]:
    """Tracks information about this model."""
    name: str
    used: bool
    result: OutT
    def __init__(self, mdl_name: str, result: OutT) -> None:
        self.name = mdl_name  # This is just the filename.
        self.used = False
        self.result = result  # Return value from compile function.

    def __repr__(self) -> str:
        return f'<Model "{self.name}, used={self.used}>'


class ModelCompiler[ModelKey: Hashable, InT, OutT]:
    """Manages the set of merged models that have been generated.

    The version number can be incremented to invalidate previous compilations.
    """
    def __init__(
        self,
        game: Game,
        studiomdl_loc: Path,
        pack: PackList,
        map_name: str,
        folder_name: str,
        version: object=0,
        pack_models: bool = True,
        compile_dir: Path | None = None,
    ) -> None:
        # The models already constructed.
        self._built_models: ACache[ModelKey, GenModel[OutT]] = ACache()

        # The random indexes we use to produce filenames.
        self._mdl_names: set[str] = set()

        self.game: Game = game
        self.model_folder = f'maps/{map_name}/{folder_name}/'
        self.model_folder_abs = game.path / 'models' / self.model_folder
        self.pack: PackList = pack
        self.version = version
        self.studiomdl_loc = studiomdl_loc
        self.compile_dir = (compile_dir / folder_name) if compile_dir is not None else None
        self.limiter = trio.CapacityLimiter(8)
        self.pack_models = pack_models
        # For statistics, the number we built this compile
        self.built_count = 0

    @classmethod
    def from_ctx(cls, ctx: Context, folder_name: str, version: object=0) -> 'ModelCompiler':
        """Convenience method to construct from the context's data."""
        if ctx.studiomdl is None:
            raise ValueError('No StudioMDL!')
        return cls(
            ctx.game,
            ctx.studiomdl,
            ctx.pack,
            ctx.bsp_path.stem,
            folder_name,
            version,
            compile_dir=ctx.modelcompile_dump,
        )

    def use_count(self) -> int:
        """Return the number of used models."""
        return sum(1 for _, mdl in self._built_models if mdl.used)

    def __enter__(self) -> Self:
        """Load the previously compiled models and prepare for compiles."""
        # Ensure the folder exists.
        os.makedirs(self.model_folder_abs, exist_ok=True)

        if force_regen:
            return self  # Skip loading.

        data: list[tuple[ModelKey, str, OutT]]
        version = 0
        try:
            with (self.model_folder_abs / 'manifest.bin').open('rb') as f:
                result = pickle.load(f)
                if isinstance(result, tuple):
                    data, version = result
                else:  # V0, no number.
                    data = result
        except FileNotFoundError:
            return self
        except Exception:
            LOGGER.warning(
                'Could not parse existing models file '
                'models/{}/manifest.bin:',
                self.model_folder,
                exc_info=True,
            )
            return self

        if version != self.version:
            # Different version, ignore the data.
            return self

        for mdl_name in self.model_folder_abs.glob('*.mdl'):
            self._mdl_names.add(str(mdl_name.stem).casefold())

        for tup in data:
            try:
                key, name, mdl_result = tup
                if not isinstance(name, str):
                    continue
            except ValueError:
                continue  # Malformed, ignore.
            if name in self._mdl_names:
                self._built_models.load(key, GenModel(name, mdl_result))
            else:
                LOGGER.warning('Model in manifest but not present: {}', name)

        LOGGER.info('Found {} existing models/{}*', len(self._built_models), self.model_folder)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Write the constructed models to the cache file and remove unused models."""
        if exc_type is not None or exc_val is not None:
            return
        data: list[tuple[ModelKey, str, OutT]] = []
        used_mdls: set[str] = set()
        for key, mdl in self._built_models:
            if mdl.used:
                data.append((key, mdl.name, mdl.result))
                used_mdls.add(mdl.name.casefold())

        with AtomicWriter(self.model_folder_abs / 'manifest.bin', is_bytes=True) as f:
            # Compatibility isn't a concern, since it'll just mean we have to
            # rebuild the models.
            pickle.dump((data, self.version), f, pickle.HIGHEST_PROTOCOL)

        culled = 0
        for mdl_file in self.model_folder_abs.glob('*'):
            if mdl_file.suffix not in {'.mdl', '.phy', '.vtx', '.vvd'}:
                continue

            # Strip all suffixes.
            if mdl_file.name[:mdl_file.name.find('.')].casefold() in used_mdls:
                continue
            culled += 1
            try:
                mdl_file.unlink()
            except FileNotFoundError:
                pass

        LOGGER.info('Culled {} models in models/{}*', culled, self.model_folder)

    async def get_model(
        self,
        key: ModelKey,
        compile_func: Callable[[ModelKey, Path, str, InT], Awaitable[OutT]],
        args: InT,
        *,
        prefix: str = 'mdl',
    ) -> tuple[str, OutT]:
        """Given a model key, either return the existing model, or compile it.

        Either way the result is the new model name, which also has been packed.
        The provided function will be called if it needs to be compiled, passing
        in the following arguments:
            * The key, used to detect if the model was compiled previously.
            * The temporary folder to write to
            * The name of the model to generate.
            * The args parameter, which can be anything. This is useful for
              passing data that can't be pickled, but the function still needs.
        It should create "mdl.qc" in the folder, and then
        StudioMDL will be called on the model to compile it. The return value will
        be passed back from this function.

        If the model key is None, a new model will always be compiled.
        The model key and return value must be pickleable, so they can be saved
        for use in subsequent compiles.

        :param prefix: Prefix in the model name, if a fresh model is created.
        """
        LOGGER.debug('Compiling model with key: {}', key)
        model = await self._built_models.fetch(
            key, ModelCompiler._compile,
            self, key, prefix, compile_func, args,
        )

        if not model.used:
            # Pack it in.
            model.used = True

            full_model_path = self.model_folder_abs / model.name
            if self.pack_models:
                LOGGER.debug('Packing model {}.mdl:', full_model_path)
                for ext in MDL_EXTS:
                    try:
                        with open(str(full_model_path.with_suffix(ext)), 'rb') as fb:
                            self.pack.pack_file(
                                f'models/{self.model_folder}{model.name}{ext}',
                                data=fb.read(),
                            )
                    except FileNotFoundError:
                        pass

        return f'models/{self.model_folder}{model.name}.mdl', model.result

    async def _compile(
        self,
        key: ModelKey,
        prefix: str,
        compile_func: Callable[[ModelKey, Path, str, InT], Awaitable[OutT]],
        args: InT,
    ) -> GenModel[OutT]:
        """Actually build the model."""
        self.built_count += 1
        # Figure out a name to use.
        while True:
            mdl_name = f'{prefix}_{random.getrandbits(16):04x}'
            if mdl_name not in self._mdl_names:
                self._mdl_names.add(mdl_name)
                break

        # If compile dir is specified, create the folder/clear it, but don't delete once done.
        ctx_man: contextlib.AbstractContextManager[str | Path]
        if self.compile_dir is not None:
            path = Path(self.compile_dir, mdl_name)
            ctx_man = contextlib.nullcontext(path)
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            path.mkdir(parents=True, exist_ok=True)
        else:  # If not specified, use a temporary directory.
            ctx_man = tempfile.TemporaryDirectory(prefix='mdl_compile')

        with ctx_man as folder:
            path = Path(folder)
            result = await compile_func(key, path, f'{self.model_folder}{mdl_name}.mdl', args)
            studio_args = [
                str(self.studiomdl_loc),
                '-nop4',
                '-game', str(self.game.path),
                str(path / 'model.qc'),
            ]
            LOGGER.debug("Execute {}", studio_args)
            async with self.limiter:
                res = await trio.run_process(studio_args, capture_stdout=True, check=False)
            if self.compile_dir is not None:
                (path / 'compile.log').write_bytes(res.stdout)
            LOGGER.debug(
                'Log for {}:\n{}',
                str(path / 'model.qc'),
                res.stdout.replace(b'\r\n', b'\n').decode('ascii', 'replace'),
            )
            res.check_returncode()  # Or raise.

        return GenModel(mdl_name, result)
