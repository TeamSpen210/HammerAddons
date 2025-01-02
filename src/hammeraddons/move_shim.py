"""Modules were previously available under the srctools package.

Implement deprecation warnings while keeping that functional.
"""
from typing import Dict, Optional, Sequence, Union
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_loader
import sys
import types
import warnings

from hammeraddons import bsp_transform, config, mdl_compiler, plugin, propcombine, props_config


moves: Dict[str, types.ModuleType] = {
    'srctools.bsp_transform': bsp_transform,
    # Loader = none means this acts like a namespace package, which is all we need.
    'srctools.compiler': module_from_spec(ModuleSpec(
        'srctools.compiler',
        None,
        is_package=True,
    )),
    'srctools.compiler.mdl_compiler': mdl_compiler,
    'srctools.compiler.propcombine': propcombine,
    'srctools.scripts.config': config,
    'srctools.props_config': props_config,
    'srctools.plugin': plugin,
}


class ModuleProxy(types.ModuleType):
    """Redirect to another module."""
    def __init__(self, orig: types.ModuleType) -> None:
        super().__init__(orig.__name__, getattr(orig, '__doc__', ''))
        super().__setattr__('_module', orig)

    def __getattr__(self, name: str) -> None:
        return getattr(super().__getattribute__('_module'), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith('__'):
            super().__setattr__(name, value)
        else:
            setattr(super().__getattribute__('_module'), name, value)


class SwapLoader(Loader):
    """When loaded redirect to the original module."""
    def __init__(self, orig: types.ModuleType) -> None:
        self.orig = orig

    def create_module(self, spec: ModuleSpec) -> Optional[types.ModuleType]:
        """Create the proxy."""
        return ModuleProxy(self.orig)

    def exec_module(self, module: types.ModuleType) -> None:
        """Do nothing."""


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
            warnings.warn(f'Import {result.__name__} instead.', DeprecationWarning, stacklevel=2)
            return spec_from_loader(
                fullname,
                SwapLoader(result),
                is_package=fullname == 'srctools.compiler',
            )


def install() -> None:
    """Install the hook."""
    sys.meta_path.insert(0, DeprecatedFinder())
