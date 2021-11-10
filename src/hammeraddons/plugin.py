"""Logic for loading all the code in arbitary locations for plugin purposes."""
import types
import sys
import operator
from pathlib import Path
from collections import deque
from importlib.util import spec_from_loader, module_from_spec
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec, SourceFileLoader
from typing import (
    Union, Optional, Set, Sequence, Tuple, Iterable,
    Iterator,
)

from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class Source:
    """A location that contains plugins."""
    def __init__(self, folder: Path, recurse: bool):
        self.folder = folder  # Folder to look in.
        self.recursive = recurse  # If we automatically load recursively or not.
        # If non-empty, load only these modules.
        self.autoload_files: Set[Path] = set()


def parse_name(prefix: str, name: str) -> Union[Tuple[int, Path], Tuple[None, None]]:
    """Parse out the source index and file path."""
    pref_size = len(prefix) + 1
    first_dot = name.find('.', pref_size)
    if name.startswith(prefix) and first_dot > 0:
        try:
            ind = int(name[pref_size:first_dot], 16)
        except ValueError:
            return None, None
        path = Path(name[first_dot + 1:].replace('.', '/'))
        return ind, path
    return None, None


def build_name(prefix: str, source_ind: int, name: Path) -> str:
    """Build a package name from the index and path."""
    if name.name.casefold() == '__init__.py':
        name = name.parent
    name = name.with_suffix('')
    dotted = str(name).replace('\\', '.').replace('/', '.')
    return f'{prefix}_{source_ind:02x}.{dotted}'


def _iter_folder(folder: Path, recursive: bool) -> Iterator[Path]:
    """Yield .py files and subfolders (packages) in a folder.

    We do a breadth-first search so parents are imported first.
    """
    folders = deque([folder])
    while folders:
        for path in folders.popleft().iterdir():
            if path.is_dir():
                package = path / '__init__.py'
                if package.exists():
                    yield package
                # Skip pycache since it doesn't have any sources.
                if recursive and path.stem != '__pycache__':
                    folders.append(path)
            # Skip init, this would be hit above.
            elif path.suffix.casefold() == '.py' and path.stem != '__init__':
                yield path


class PluginRootLoader(Loader):
    """Fake loader to create the pseduo-module all plugins are installed into."""
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def create_module(self, spec: ModuleSpec) -> Optional[types.ModuleType]:
        """Use the default behaviour."""
        return None


class PluginFinder(MetaPathFinder):
    """Loads plugins.

    Plugins
    """
    def __init__(self, prefix: str, sources: Iterable[Source]) -> None:
        self.prefix = prefix
        self.sources = sorted(sources, key=operator.attrgetter('folder'))

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[Union[str, bytes]]],
        target: Optional[types.ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        """Load a module."""
        source_ind, subpath = parse_name(self.prefix, fullname)
        if source_ind is None or subpath is None or source_ind < 0:
            return None
        try:
            source = self.sources[source_ind]
        except IndexError:
            return None
        new_path = source.folder / subpath
        filename = str(new_path / '__init__.py') if new_path.is_dir() else str(new_path)
        loader = SourceFileLoader(fullname, filename)
        return spec_from_loader(fullname, loader)

    def load_all(self) -> None:
        """Load all the plugin modules."""
        # Inject our prefix as a pseudo-module, so it can be imported.
        sys.modules[self.prefix] = module = module_from_spec(ModuleSpec(
            self.prefix,
            PluginRootLoader(self.prefix + '.py'),
            is_package=True,
        ))
        module.__doc__ = 'Plugin pseduo-package.'
        for i, source in enumerate(self.sources):
            source_modname = f'{self.prefix}_{i:02x}'
            sys.modules[source_modname] = module = module_from_spec(ModuleSpec(
                source_modname,
                PluginRootLoader(source_modname + '.py'),
                is_package=True,
            ))
            module.__doc__ = 'Plugin pseduo-package.'

            paths: Iterable[Path]
            if source.autoload_files:
                paths = source.autoload_files
            else:
                paths = _iter_folder(source.folder, source.recursive)
            for path in paths:
                name = build_name(self.prefix, i, path.relative_to(source.folder))
                if name in sys.modules:
                    # Already loaded, by user import.
                    LOGGER.info('Plugin "{}" was preloaded automatically.', name)
                    continue
                LOGGER.info('Loading "{}" as "{}"', path, name)

                filename = str(path / '__init__.py') if path.is_dir() else str(path)
                loader = SourceFileLoader(name, filename)
                spec = spec_from_loader(name, loader)
                sys.modules[name] = module = module_from_spec(spec)

                # Provide a logger for the plugin, already setup.
                setattr(module, 'LOGGER', get_logger(name, "plugin:" + str(path)))
                loader.exec_module(module)
