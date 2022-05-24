"""Logic for loading all the code in arbitary locations for plugin purposes."""
import types
import sys
from pathlib import Path
from collections import deque
from importlib.util import spec_from_loader, module_from_spec
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec, SourceFileLoader
from typing import Dict, Union, Optional, Set, Sequence, Tuple, Iterable, Iterator
from typing_extensions import Final

import attrs

from srctools import Property
from srctools.logger import get_logger

LOGGER = get_logger(__name__)
BUILTIN: Final = 'builtin'


@attrs.define
class Source:
    """A location that contains plugins."""
    id: str
    folder: Path  # Folder to look in.
    recursive: bool = False  # If we automatically load recursively or not.
    # If non-empty load only these modules.
    files: Set[Path] = attrs.Factory(set)

    @classmethod
    def parse(cls, root: Path, prop: Property) -> 'Source':
        """Parse from a property."""
        if prop.has_children():
            if not prop.real_name:
                raise ValueError('Plugins must have a unique ID!')
            if not prop.real_name.isidentifier() or '.' in prop.real_name:
                raise ValueError(
                    f'Plugin names must be a valid identifier '
                    f'(words, numbers, underscore), not "{prop.real_name}"!'
                )
            try:
                path = (root / Path(prop['path'])).resolve()
            except LookupError:
                raise ValueError(f'Plugin "{prop.real_name}" must have a path!') from None
            if path.is_dir():
                return Source(
                    prop.real_name,
                    path,
                    recursive=prop.bool('recurse'),
                )
            else:
                return Source(
                    prop.real_name,
                    path.parent,
                    files={path},
                )
        else:  # Legacy style.
            LOGGER.warning('Plugins should use block definition style!')
            path = (root / Path(prop.value)).resolve()
            if prop.name in ('path', "recursive", 'folder'):
                if not path.is_dir():
                    raise ValueError(f"'{path}' is not a directory!")

                return Source('', path, prop.name == "recursive")
            elif prop.name in ('single', 'file'):
                return Source('', path.parent, files={path})
            elif prop.name == '_builtin_':
                return Source(BUILTIN, path, recursive=True)
            else:
                raise ValueError("Unknown plugins key {}".format(prop.real_name))


def parse_name(prefix: str, name: str) -> Union[Tuple[str, Path], Tuple[None, None]]:
    """Parse out the source index and file path."""
    pref_size = len(prefix) + 1
    first_dot = name.find('.', pref_size)
    if name.startswith(prefix) and first_dot > 0:
        source = name[pref_size:first_dot]
        path = Path(name[first_dot + 1:].replace('.', '/'))
        return source, path
    return None, None


def build_name(prefix: str, source: str, name: Path) -> str:
    """Build a package name from the index and path."""
    if name.name.casefold() == '__init__.py':
        name = name.parent
    name = name.with_suffix('')
    dotted = str(name).replace('\\', '.').replace('/', '.')
    return f'{prefix}.{source}.{dotted}'


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
    """Loads plugins."""
    def __init__(self, prefix: str, sources: Dict[str, Source]) -> None:
        self.prefix = prefix
        self.sources = sources

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[Union[str, bytes]]],
        target: Optional[types.ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        """Load a module."""
        source_id, subpath = parse_name(self.prefix, fullname)
        if source_id is None or subpath is None:
            return None
        try:
            source = self.sources[source_id]
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
        for source in self.sources.values():
            source_modname = f'{self.prefix}.{source.id}'
            sys.modules[source_modname] = module = module_from_spec(ModuleSpec(
                source_modname,
                PluginRootLoader(source_modname + '.py'),
                is_package=True,
            ))
            module.__doc__ = 'Plugin pseduo-package.'

        for source in self.sources.values():
            paths: Iterable[Path]
            if source.files:
                paths = source.files
            else:
                paths = _iter_folder(source.folder, source.recursive)
            for path in paths:
                name = build_name(self.prefix, source.id, path.relative_to(source.folder))
                if name in sys.modules:
                    # Already loaded, by user import.
                    LOGGER.info('Plugin "{}" was preloaded automatically.', name)
                    continue
                LOGGER.info('Loading "{}" as "{}"', path, name)

                filename = str(path / '__init__.py') if path.is_dir() else str(path)
                loader = SourceFileLoader(name, filename)
                spec = spec_from_loader(name, loader)
                if spec is None:
                    raise AssertionError(f'No spec for {name!r}!')
                sys.modules[name] = module = module_from_spec(spec)

                # Provide a logger for the plugin, already setup.
                setattr(module, 'LOGGER', get_logger(name, f"plugin.{source.id}:{path}"))
                loader.exec_module(module)
