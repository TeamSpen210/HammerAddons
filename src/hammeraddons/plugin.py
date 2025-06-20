"""Logic for loading all the code in arbitary locations for plugin purposes."""
from typing import Final, Self
from collections.abc import Callable, Iterable, Iterator, Sequence
from collections import deque
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import spec_from_loader
from pathlib import Path
import importlib
import types

from srctools import Keyvalues
from srctools.logger import get_logger
import attrs


LOGGER = get_logger(__name__)
BUILTIN: Final = 'builtin'


@attrs.define
class Source:
    """A location that contains plugins."""
    id: str
    folder: Path  # Folder to look in.
    recursive: bool = False  # If we automatically load recursively or not.
    # If non-empty load only these modules.
    files: set[Path] = attrs.Factory(set)

    @classmethod
    def parse(cls, kv: Keyvalues, path_parse: Callable[[str], Path]) -> Self:
        """Parse from keyvalues data."""
        if kv.has_children():
            if not kv.real_name:
                raise ValueError('Plugins must have a unique ID!')
            if not kv.real_name.isidentifier() or '.' in kv.real_name:
                raise ValueError(
                    f'Plugin names must be a valid identifier '
                    f'(words, numbers, underscore), not "{kv.real_name}"!'
                )
            try:
                path = path_parse(kv['path'])
            except LookupError:
                raise ValueError(f'Plugin "{kv.real_name}" must have a path!') from None
            if path.is_dir():
                return cls(
                    kv.real_name,
                    path,
                    recursive=kv.bool('recurse'),
                )
            else:
                return cls(
                    kv.real_name,
                    path.parent,
                    files={path},
                )
        else:  # Legacy style.
            LOGGER.warning('Plugins should use block definition style!')
            path = path_parse(kv.value)
            if kv.name in ('path', "recursive", 'folder'):
                if not path.is_dir():
                    raise ValueError(f"'{path}' is not a directory!")

                return cls('', path, kv.name == "recursive")
            elif kv.name in ('single', 'file'):
                return cls('', path.parent, files={path})
            elif kv.name == '_builtin_':
                return cls(BUILTIN, path, recursive=True)
            else:
                raise ValueError(f"Unknown plugins key {kv.real_name}")


def parse_name(prefix: str, name: str) -> tuple[str, Path] | tuple[None, None]:
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


class PluginFinder(MetaPathFinder):
    """Loads plugins."""
    def __init__(self, prefix: str, sources: dict[str, Source]) -> None:
        self.prefix = prefix
        self.sources = sources
        # All names in a package hierarchy need to exist, so we need to produce a module for
        # each source folder, and the root prefix. Using loader=None here gives a namespace
        # package which is all we need.
        self.ns_packages = {
            (name := f'{self.prefix}.{source.id}'): ModuleSpec(name, None, is_package=True)
            for source in self.sources.values()
        }
        self.ns_packages[prefix] = ModuleSpec(prefix, None, is_package=True)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str | bytes] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Locate the module spec for a module, if it's one of ours."""
        # First check the various roots, and return the namespace package if so.
        try:
            return self.ns_packages[fullname]
        except KeyError:
            pass
        source_id, subpath = parse_name(self.prefix, fullname)
        if source_id is None or subpath is None:
            return None
        try:
            source = self.sources[source_id]
        except IndexError:
            return None
        new_path = source.folder / subpath
        if new_path.is_dir():
            new_path /= '__init__.py'
        else:
            new_path = new_path.with_suffix('.py')
        # SourceFileLoader can do all the hard work for us given a source file.
        return spec_from_loader(fullname, SourceFileLoader(fullname, str(new_path)))

    def load_all(self) -> None:
        """Load all the plugin modules."""
        for source in self.sources.values():
            paths: Iterable[Path]
            if source.files:
                paths = source.files
            else:
                paths = _iter_folder(source.folder, source.recursive)
            for path in paths:
                name = build_name(self.prefix, source.id, path.relative_to(source.folder))
                LOGGER.info('Loading "{}" as "{}"', path, name)
                # Do an import, which will call back to find_spec().
                importlib.import_module(name)
