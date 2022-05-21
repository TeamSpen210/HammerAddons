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


moves: Dict[str, ModuleSpec] = {
    'srctools.bsp_transform': bsp_transform.__spec__,
    # Loader = none means this acts like a namespace package, which is all we need.
    'srctools.compiler': ModuleSpec(
        'srctools.compiler',
        None,
        is_package=True,
    ),
    'srctools.compiler.mdl_compiler': mdl_compiler.__spec__,
    'srctools.compiler.propcombine': propcombine.__spec__,
    'srctools.scripts.config': config.__spec__,
    'srctools.props_config': props_config.__spec__,
    'srctools.plugin': plugin.__spec__,
}


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
