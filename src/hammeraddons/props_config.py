"""Keyvalues-based configuration system.

A list of options are passed in, which parse each option to a basic type.
"""
from typing import IO, Dict, Generic, Iterable, List, Optional, Type, TypeVar, Union, overload
from typing_extensions import TypeAlias
from pathlib import Path
import inspect

from srctools import Keyvalues, Vec, conv_bool, parse_vec_str
from srctools.logger import get_logger
import attrs


LOGGER = get_logger(__name__)


Option: TypeAlias = Union[str, int, float, bool, Vec, Keyvalues]
OptionT = TypeVar('OptionT', Keyvalues, str, int, float, bool, Vec)

TYPE_NAMES: Dict[Type[Option], str] = {
    str: 'Text',
    int: 'Whole Number',
    float: 'Decimal Number',
    bool: 'True/False',
    Vec: 'Vector',
    Keyvalues: 'Keyvalues Block',
}


@attrs.define(init=False)
class Opt(Generic[OptionT]):
    """A type of option that can be chosen.
    """
    id: str
    name: str
    kind: Type[OptionT]
    fallback: Optional[str]
    doc: List[str]

    def __init__(
        self,
        opt_id: str,
        kind: Type[OptionT],
        doc: str,
        fallback: Optional[str],
    ) -> None:
        self.kind = kind
        self.id = opt_id.casefold()
        self.name = opt_id
        self.fallback = fallback
        # Remove indentation, and trailing carriage return
        self.doc = inspect.cleandoc(doc).rstrip().splitlines()

    @classmethod
    def block(
        cls,
        opt_id: str,
        default: Keyvalues,
        doc: str, *,
        fallback: str = None,
    ) -> 'OptWithDefault[Keyvalues]':
        """Return an option giving the raw keyvalues block.

        These always use an empty block as the default.
        """
        return OptWithDefault(opt_id, Keyvalues, default.copy(), doc, fallback)

    @classmethod
    def string_or_none(cls, opt_id: str, doc: str, *, fallback: str = None) -> 'Opt[str]':
        """Return a string-type option, with no default."""
        return Opt(opt_id, str, doc, fallback)

    @classmethod
    def boolean_or_none(cls, opt_id: str, doc: str, *, fallback: str = None) -> 'Opt[bool]':
        """Return a boolean-type option, with no default."""
        return Opt(opt_id, bool, doc, fallback)

    @classmethod
    def integer_or_none(cls, opt_id: str, doc: str, *, fallback: str = None) -> 'Opt[int]':
        """Return an integer-type option, with no default."""
        return Opt(opt_id, int, doc, fallback)

    @classmethod
    def floating_or_none(cls, opt_id: str, doc: str, *, fallback: str = None) -> 'Opt[float]':
        """Return a float-type option, with no default."""
        return Opt(opt_id, float, doc, fallback)

    @classmethod
    def vector_or_none(cls, opt_id: str, doc: str, *, fallback: str = None) -> 'Opt[Vec]':
        """Return a vector-type option, with no default."""
        return Opt(opt_id, Vec, doc, fallback)

    @classmethod
    def string(cls, opt_id: str, default: str, doc: str, *, fallback: str = None) -> 'OptWithDefault[str]':
        """Return a string-type option."""
        return OptWithDefault(opt_id, str, default, doc, fallback)

    @classmethod
    def boolean(cls, opt_id: str, default: bool, doc: str, *, fallback: str = None) -> 'OptWithDefault[bool]':
        """Return a boolean-type option."""
        return OptWithDefault(opt_id, bool, default, doc, fallback)

    @classmethod
    def integer(cls, opt_id: str, default: int, doc: str, *, fallback: str = None) -> 'OptWithDefault[int]':
        """Return an integer-type option."""
        return OptWithDefault(opt_id, int, default, doc, fallback)

    @classmethod
    def floating(cls, opt_id: str,  default: float, doc: str, *, fallback: str = None) -> 'OptWithDefault[float]':
        """Return a float-type option."""
        return OptWithDefault(opt_id, float, default, doc, fallback)

    @classmethod
    def vector(cls, opt_id: str, default: Vec,  doc: str, *, fallback: str = None) -> 'OptWithDefault[Vec]':
        """Return a vector-type option."""
        return OptWithDefault(opt_id, Vec, default, doc, fallback)


@attrs.define(init=False)
class OptWithDefault(Opt[OptionT], Generic[OptionT]):
    """An option, with a default."""
    default: OptionT
    def __init__(
        self,
        opt_id: str,
        kind: Type[OptionT],
        default: OptionT,
        doc: str,
        fallback: Optional[str],
    ) -> None:
        super().__init__(opt_id, kind, doc, fallback)  # type: ignore
        self.default = default
        if fallback is not None:
            self.doc.append(f'If unset, the default is read from `{default}`.')


class Options:
    """Allows parsing a set of Keyvalues option blocks."""
    defaults: List[Opt]
    settings: Dict[str, Union[None, str, int, float, bool, Vec, Keyvalues]]
    path: Optional[Path]

    def __init__(self, defaults: Union[Iterable[Opt], dict]) -> None:
        if isinstance(defaults, dict):
            self.defaults = [
                opt for opt in defaults.values()
                if isinstance(opt, Opt)
            ]
        else:
            self.defaults = list(defaults)

        self.settings = {}
        self.path = None

    def load(self, opt_blocks: Keyvalues) -> None:
        """Read settings from the given keyvalues block."""
        self.settings.clear()
        set_vals = {}
        for opt_block in opt_blocks:
            for prop in opt_block:
                set_vals[prop.name] = prop

        options: Dict[str, Opt] = {opt.id: opt for opt in self.defaults}
        if len(options) != len(self.defaults):
            from collections import Counter

            # Find ids used more than once.
            raise Exception('Duplicate option(s)! ({})'.format(', '.join(
                k for k, v in
                Counter(opt.id for opt in self.defaults).items()
                if v > 1
            )))

        fallback_opts = []

        for opt in self.defaults:
            if isinstance(opt, OptWithDefault):
                default = opt.default
            else:
                default = None

            try:
                prop = set_vals.pop(opt.id)
            except KeyError:
                if opt.fallback is not None:
                    fallback_opts.append(opt)
                    assert opt.fallback in options, 'Invalid fallback in ' + opt.id
                else:
                    self.settings[opt.id] = default
                continue
            if opt.kind is Keyvalues:
                self.settings[opt.id] = prop.copy()
                continue

            # Non-RAW types cannot have a keyvalues block, only a value.
            if prop.has_children():
                raise ValueError(f'Cannot use keyvalues block for "{opt.name}"')

            if opt.kind is Vec:
                # Pass nones to allow us to check if it failed.
                x, y, z = parse_vec_str(prop.value, x=None)
                if x is None:
                    self.settings[opt.id] = default.copy() if default is not None else None
                else:
                    self.settings[opt.id] = Vec(x, y, z)
            elif opt.kind is bool:
                self.settings[opt.id] = conv_bool(prop.value, default)
            else:  # int, float, str - no special handling...
                try:
                    self.settings[opt.id] = opt.kind(prop.value)
                except (ValueError, TypeError):
                    self.settings[opt.id] = default

        for opt in fallback_opts:
            assert opt.fallback is not None
            try:
                self.settings[opt.id] = self.settings[opt.fallback]
            except KeyError:
                raise Exception(f'Bad fallback for "{opt.id}"!')
            # Check they have the same type.
            if opt.kind is not options[opt.fallback].kind:
                raise ValueError(
                    f'"{opt.id}" cannot fall back to "{opt.fallback}" - different type!'
                )

        if set_vals:
            LOGGER.warning('Extra config options: {}', set_vals)

    def set_opt(self, option: Opt[OptionT], value: OptionT) -> None:
        """Set an option to a specific value."""
        if option.id not in self.settings:
            LOGGER.warning('Invalid option "{}"!', option.name)
            return

        if type(value) is not option.kind:
            raise ValueError(f'Value "{value!r}" is not the same as option "{option.name}": {option.kind}')
        else:
            self.settings[option.id] = value

    @overload
    def get(self, option: OptWithDefault[OptionT]) -> OptionT: ...
    @overload
    def get(self, option: Opt[OptionT]) -> Optional[OptionT]: ...

    def get(self, option: Opt[OptionT]) -> Optional[OptionT]:
        """Fetch the given option, or return None if not present and no default is defined."""
        try:
            val = self.settings[option.id]
        except KeyError:
            raise TypeError(f'Option "{option.name}" does not exist!') from None

        if val is None:
            if option.kind is Keyvalues:  # Type checker doesn't understand isinstance here.
                return Keyvalues(option.name, [])  # type: ignore
            else:
                return None

        # Don't allow subclasses (bool/int)
        if type(val) is not option.kind:
            raise ValueError(f'Option "{option.name}" is {type(val)} (code expected {option.kind})')

        # Vec is mutable, don't allow modifying the original.
        if option.kind is Vec or option.kind is Keyvalues:
            assert isinstance(val, Vec) or isinstance(val, Keyvalues)
            return val.copy()
        else:
            assert isinstance(val, option.kind)
            return val

    def save(self, file: IO[str]) -> None:
        """Write the current config out to the given file.

        Descriptions are written out as comments.
        """
        file.write('"Config"\n\t{\n')
        for ind, option in enumerate(self.defaults):
            if ind != 0:
                file.write('\n\n')
            for line in option.doc:
                file.write('\t// {}\n'.format(line))

            if isinstance(option, OptWithDefault):
                default = option.default
            else:
                default = None

            # PROP types are "raw", so they don't have defaults.
            if option.kind is not Keyvalues and isinstance(option, OptWithDefault):
                if isinstance(default, bool):
                    default = '1' if default else '0'

                file.write(f'\t// Default Value: "{default}"\n')

            try:
                value = self.settings[option.id]
            except KeyError:
                value = default

            if isinstance(value, bool):
                value = '1' if value else '0'

            if value is None:
                # Comment out the unset value.
                file.write(f'\t// "{option.name}" ""\n')
            elif isinstance(value, Keyvalues):
                value.name = option.name
                for line in value.export():
                    file.write('\t' + line)
            else:
                file.write(f'\t"{option.name}" "{value}"\n')
        file.write('\t}\n')
