"""Manages potential models that are being generated.

Each comes with a key, used to identify a previously compiled version.
We can then reuse already compiled versions.
"""
import os
import pickle
import subprocess
import tempfile
import random
from typing import Dict, Callable, TypeVar, Tuple, Set, List, Optional, Hashable
from pathlib import Path, PurePosixPath

from srctools import AtomicWriter
from srctools.bsp_transform import Context
from srctools.game import Game
from srctools.mdl import MDL_EXTS
from srctools.packlist import PackList, LOGGER


ModelKey = TypeVar('ModelKey', bound=Hashable)


class GenModel:
    """Tracks information about this model."""
    def __init__(self, mdl_name: str, used: bool = False) -> None:
        self.name = mdl_name  # This is just the filename.
        self.used = used

    def __repr__(self) -> str:
        return f'<Model "{self.name}, used={self.used}>'


class ModelCompiler:
    """Manages the set of merged models that have been generated."""
    def __init__(
        self,
        game: Game,
        studiomdl_loc: Optional[Path],
        pack: PackList,
        map_name: str,
        folder_name: str,
    ) -> None:
        # The models already constructed.
        self._built_models: Dict[ModelKey, GenModel] = {}

        # The random indexes we use to produce filenames.
        self._mdl_names: Set[str] = set()

        self.game = game
        self.model_folder = 'maps/{}/{}/'.format(map_name, folder_name)
        self.model_folder_abs = game.path / 'models' / self.model_folder
        self.pack = pack

        if studiomdl_loc is None:
            studiomdl_loc = game.bin_folder() / 'studiomdl.exe'
        self.studiomdl_loc = studiomdl_loc.resolve()

    @classmethod
    def from_ctx(cls, ctx: Context, folder_name: str) -> 'ModelCompiler':
        """Convenience method to construct from the context's data."""
        return cls(
            ctx.game,
            ctx.studiomdl,
            ctx.pack,
            ctx.bsp_path.stem,
            folder_name,
        )

    def use_count(self) -> int:
        """Return the number of used models."""
        return sum(1 for mdl in self._built_models.values() if mdl.used)

    def __enter__(self) -> 'ModelCompiler':
        # Ensure the folder exists.
        os.makedirs(self.model_folder, exist_ok=True)
        try:
            with (self.model_folder_abs / 'manifest.bin').open('rb') as f:
                data: List[Tuple[ModelKey, str]] = pickle.load(f)
                assert isinstance(data, list)
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

        for mdl_name in self.model_folder_abs.glob('*.mdl'):
            self._mdl_names.add(str(mdl_name.stem).casefold())

        for key, name in data:
            if name in self._mdl_names:
                self._built_models[key] = GenModel(name)
            else:
                LOGGER.warning('Model in manifest but not present: {}', name)

        LOGGER.info('Found {} existing models/{}*', len(self._built_models), self.model_folder)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Write the constructed models to the cache file and remove unused models."""
        if exc_type is not None or exc_val is not None:
            return False
        data = []
        used_mdls = set()
        for key, mdl in self._built_models.items():
            if mdl.used:
                data.append((key, mdl.name))
                used_mdls.add(mdl.name.casefold())

        with AtomicWriter(self.model_folder_abs / 'manifest.bin', is_bytes=True) as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        for mdl_file in self.model_folder_abs.glob('*'):
            if mdl_file.suffix not in {'.mdl', '.phy', '.vtx', '.vvd'}:
                continue

            # Strip all suffixes.
            if mdl_file.name[:mdl_file.name.find('.')].casefold() in used_mdls:
                continue

            LOGGER.info('Culling {}...', mdl_file)
            try:
                mdl_file.unlink()
            except FileNotFoundError:
                pass

    def get_model(
        self,
        key: ModelKey,
        compile_func: Callable[[ModelKey, Path, str], None],
    ) -> str:
        """Given a model key, either return the existing model, or compile it.

        Either way the result is the new model name, which also has been packed.
        The provided function will be called if it needs to be compiled, passing
        in the following arguments:
            * The key
            * the temporary folder to write to
            * the name of the model to generate.
        It should create "mdl.qc" in the folder, and then
        StudioMDL will be called on the model to comile it.

        If the model key is None, a new model will always be compiled.
        """
        try:
            model = self._built_models[key]
        except KeyError:
            # Need to build the model.
            # Figure out a name to use.
            while True:
                mdl_name = 'mdl_{:04x}'.format(random.getrandbits(16))
                if mdl_name not in self._mdl_names:
                    self._mdl_names.add(mdl_name)
                    break

            model = self._built_models[key] = GenModel(mdl_name)

            with tempfile.TemporaryDirectory(prefix='mdl_compile') as folder:
                path = Path(folder)
                compile_func(key, path, f'{self.model_folder}{mdl_name}.mdl')
                args = [
                    str(self.studiomdl_loc),
                    '-nop4',
                    '-game', str(self.game.path),
                    str(path / 'model.qc'),
                ]
                res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                LOGGER.debug(
                    'Executing {}:\n{}',
                    args,
                    res.stdout.replace(b'\r\n', b'\n').decode('ascii', 'replace'),
                )
                res.check_returncode()

        if not model.used:
            # Pack it in.
            model.used = True

            full_model_path = self.model_folder_abs / model.name
            LOGGER.debug('Packing model {}.mdl:', full_model_path)
            for ext in MDL_EXTS:
                try:
                    with open(str(full_model_path.with_suffix(ext)), 'rb') as fb:
                        self.pack.pack_file(
                            'models/{}{}{}'.format(
                                self.model_folder, model.name, ext,
                            ),
                            data=fb.read(),
                        )
                except FileNotFoundError:
                    pass

        return f'models/{self.model_folder}{model.name}.mdl'
