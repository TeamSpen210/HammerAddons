"""Modules were previously available under the srctools package.

Implement deprecation warnings while keeping that functional.
"""
from typing import Optional, Union, Sequence, Dict
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
import sys
import types
import warnings

from hammeraddons import bsp_transform, mdl_compiler, plugin, propcombine, config, props_config


def mod_to_spec(mod: types.ModuleType) -> ModuleSpec:
    """Fetch the module's __spec__."""
    spec = mod.__spec__
    assert spec is not None, mod
    return spec


moves: Dict[str, ModuleSpec] = {
    'srctools.bsp_transform': mod_to_spec(bsp_transform),
    # Loader = none means this acts like a namespace package, which is all we need.
    'srctools.compiler': ModuleSpec(
        'srctools.compiler',
        None,
        is_package=True,
    ),
    'srctools.compiler.mdl_compiler': mod_to_spec(mdl_compiler),
    'srctools.compiler.propcombine': mod_to_spec(propcombine),
    'srctools.scripts.config': mod_to_spec(config),
    'srctools.props_config': mod_to_spec(props_config),
    'srctools.plugin': mod_to_spec(plugin),
}
del mod_to_spec


class DeprecatedFinder(MetaPathFinder):
    """If it's one of ours, return the original module loader."""
    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[Union[bytes, str]]],
        target: Optional[types.ModuleType]=None,
    ) -> Optional[ModuleSpec]:
        """If known, return the required spec."""
        try:
            result = moves[fullname]
        except KeyError:
            return None
        else:
            warnings.warn(f'Import {result.name} instead.', DeprecationWarning, stacklevel=2)
            return result


def install() -> None:
    """Install the hook."""
    sys.meta_path.insert(0, DeprecatedFinder())
