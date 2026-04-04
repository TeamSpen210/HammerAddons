"""Keyvalues-based configuration system.

A list of options are passed in, which parse each option to a basic type.
"""
from typing import Any, IO, overload, Protocol, assert_never
from collections.abc import Iterable, Buffer
from pathlib import Path
import inspect
import struct

from srctools import Keyvalues, Vec, conv_bool, parse_vec_str
from srctools.logger import get_logger
import attrs


LOGGER = get_logger(__name__)


type Option = str | int | float | bool | Vec | Keyvalues

TYPE_NAMES: dict[type[Option], str] = {
    str: 'Text',
    int: 'Whole Number',
    float: 'Decimal Number',
    bool: 'True/False',
    Vec: 'Vector',
    Keyvalues: 'Keyvalues Block',
}
# Unique byte for hashing.
TYPE_BYTE: dict[type[object], bytes] = {
    typ: bytes([ind]) for ind, typ in enumerate(TYPE_NAMES)
}


class Hasher(Protocol):
    """A hashlib hash object."""
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def update(self, data: Buffer, /) -> None: ...


@attrs.define(init=False)
class Opt[OptionT: Option]:
    """A type of option that can be chosen.
    """
    id: str
    name: str
    kind: type[OptionT]
    deprecated: bool  # Deprecated, don't write out in new configs if unset.
    fallback: str | None  # If not set, copy from this other option.
    doc: list[str]

    def __init__(
        self,
        opt_id: str,
        kind: type[OptionT],
        doc: str,
        fallback: str | None,
        deprecated: bool,
    ) -> None:
        self.kind = kind
        self.id = opt_id.casefold()
        self.name = opt_id
        self.fallback = fallback
        self.deprecated = deprecated
        # Remove indentation, and trailing carriage return
        self.doc = inspect.cleandoc(doc).rstrip().splitlines()

    @classmethod
    def block(
        cls,
        opt_id: str,
        default: Keyvalues,
        doc: str, *,
        fallback: str | None = None,
        deprecated: bool = False,
    ) -> 'OptWithDefault[Keyvalues]':
        """Return an option giving the raw keyvalues block.

        These always use an empty block as the default.
        """
        return OptWithDefault(opt_id, Keyvalues, default.copy(), doc, fallback, deprecated)

    @classmethod
    def string_or_none(
        cls, opt_id: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'Opt[str]':
        """Return a string-type option, with no default."""
        return Opt(opt_id, str, doc, fallback, deprecated)

    @classmethod
    def boolean_or_none(
        cls, opt_id: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'Opt[bool]':
        """Return a boolean-type option, with no default."""
        return Opt(opt_id, bool, doc, fallback, deprecated)

    @classmethod
    def integer_or_none(
        cls, opt_id: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'Opt[int]':
        """Return an integer-type option, with no default."""
        return Opt(opt_id, int, doc, fallback, deprecated)

    @classmethod
    def floating_or_none(
        cls, opt_id: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'Opt[float]':
        """Return a float-type option, with no default."""
        return Opt(opt_id, float, doc, fallback, deprecated)

    @classmethod
    def vector_or_none(
        cls, opt_id: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'Opt[Vec]':
        """Return a vector-type option, with no default."""
        return Opt(opt_id, Vec, doc, fallback, deprecated)

    @classmethod
    def string(
        cls, opt_id: str, default: str, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'OptWithDefault[str]':
        """Return a string-type option."""
        return OptWithDefault(opt_id, str, default, doc, fallback, deprecated)

    @classmethod
    def boolean(
        cls, opt_id: str, default: bool, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'OptWithDefault[bool]':
        """Return a boolean-type option."""
        return OptWithDefault(opt_id, bool, default, doc, fallback, deprecated)

    @classmethod
    def integer(
        cls, opt_id: str, default: int, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'OptWithDefault[int]':
        """Return an integer-type option."""
        return OptWithDefault(opt_id, int, default, doc, fallback, deprecated)

    @classmethod
    def floating(
        cls, opt_id: str, default: float, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'OptWithDefault[float]':
        """Return a float-type option."""
        return OptWithDefault(opt_id, float, default, doc, fallback, deprecated)

    @classmethod
    def vector(
        cls, opt_id: str, default: Vec, doc: str, *,
        fallback: str | None = None, deprecated: bool = False,
    ) -> 'OptWithDefault[Vec]':
        """Return a vector-type option."""
        return OptWithDefault(opt_id, Vec, default, doc, fallback, deprecated)

    def hash(self, digest: Hasher) -> None:
        """Add in the state of this config."""
        digest.update(TYPE_BYTE[self.kind])
        digest.update(self.id.encode('utf8', 'replace'))
        if self.fallback is not None:
            digest.update(b'\xF0' + self.fallback.encode('utf8', 'replace'))


@attrs.define(init=False)  # __attrs_init__() is incompatible with the superclass.
class OptWithDefault[OptionT: Option](Opt[OptionT]):  # type: ignore[override]
    """An option, with a default."""
    default: OptionT

    def __init__(
        self,
        opt_id: str,
        kind: type[OptionT],
        default: OptionT,
        doc: str,
        fallback: str | None,
        deprecated: bool,
    ) -> None:
        super().__init__(opt_id, kind, doc, fallback, deprecated)
        self.default = default
        if fallback is not None:
            self.doc.append(f'If unset, the default is read from `{default}`.')

    def hash(self, digest: Hasher) -> None:
        """Include the default value."""
        super().hash(digest)
        match self.default:
            case str() as text:
                digest.update(b'\xF1' + text.encode('utf8', 'replace'))
            case int() as ordinal:
                digest.update(struct.pack('<Bq', 0xF2, ordinal))
            case float() as number:
                digest.update(struct.pack('<Bd', 0xF3, number))
            case False:
                digest.update(b'\xF4')
            case True:
                digest.update(b'\xF5')
            case Vec(x, y, z):
                digest.update(struct.pack('<Bddd', 0xF6, x, y, z))
            case Keyvalues() as kv:
                digest.update(b'\xF7' + kv.serialise().encode('utf8', 'replace'))
            case never:
                assert_never(never)


class Options:
    """Allows parsing a set of Keyvalues option blocks."""
    version: int
    defaults: list[Opt]
    settings: dict[str, Option | None]
    path: Path | None

    def __init__(self, name: str, version: int, defaults: Iterable[Opt] | dict[Any, Opt]) -> None:
        if isinstance(defaults, dict):
            self.defaults = [
                opt for opt in defaults.values()
                if isinstance(opt, Opt)
            ]
        else:
            self.defaults = list(defaults)

        self.settings = {}
        self.name = name
        self.version = version
        self.path = None

    def hash(self, digest: Hasher) -> None:
        """Add in the shape of this config."""
        digest.update(struct.pack('<IH', self.version, len(self.defaults)))
        for opt in self.defaults:
            opt.hash(digest)

    def load(self, keyvalues: Keyvalues) -> None:
        """Read settings from the given keyvalues block."""
        self.settings.clear()
        set_vals = {}
        for child in keyvalues:
            set_vals[child.name] = child

        options: dict[str, Opt] = {opt.id: opt for opt in self.defaults}
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
                raise Exception(f'Bad fallback "{opt.fallback}" for "{opt.id}"!') from None
            # Check they have the same type.
            if opt.kind is not options[opt.fallback].kind:
                raise ValueError(
                    f'"{opt.id}" cannot fall back to "{opt.fallback}" - different type!'
                )

        if set_vals:
            LOGGER.warning('Extra config options: {}', set_vals)

    def set_opt[OptionT: Option](self, option: Opt[OptionT], value: OptionT) -> None:
        """Set an option to a specific value."""
        if option.id not in self.settings:
            LOGGER.warning('Invalid option "{}"!', option.name)
            return

        if type(value) is not option.kind:
            raise ValueError(f'Value "{value!r}" is not the same as option "{option.name}": {option.kind}')
        else:
            self.settings[option.id] = value

    @overload
    def get[OptionT: Option](self, option: OptWithDefault[OptionT]) -> OptionT: ...
    @overload
    def get[OptionT: Option](self, option: Opt[OptionT]) -> OptionT | None: ...

    def get[OptionT: Option](self, option: Opt[OptionT]) -> Option | None:
        """Fetch the given option, or return None if not present and no default is defined."""
        if option.deprecated:
            raise ValueError('\n'.join(['Option was removed:', *option.doc]))
        try:
            val = self.settings[option.id]
        except KeyError:
            raise TypeError(f'Option "{option.name}" does not exist!') from None

        if val is None:
            if option.kind is Keyvalues:
                return Keyvalues(option.name, [])
            else:
                return None

        # Don't allow subclasses (bool/int).
        if type(val) is not option.kind:
            raise ValueError(f'Option "{option.name}" is {type(val)} (code expected {option.kind})')

        # Vec is mutable, don't allow modifying the original.
        if option.kind is Vec or option.kind is Keyvalues:
            assert isinstance(val, Vec) or isinstance(val, Keyvalues)
            return val.copy()
        else:
            assert isinstance(val, option.kind)
            return val

    def save(self, file: IO[str], block_name: str) -> None:
        """Write the current config out to the given file.

        Descriptions are written out as comments.
        """
        file.write(f'"{block_name}"\n\t{{\n')
        has_previous = False
        for option in self.defaults:
            if isinstance(option, OptWithDefault):
                default = option.default
            else:
                default = None

            try:
                value = self.settings[option.id]
            except KeyError:
                value = default
            if value is default and option.deprecated:
                # Never set and deprecated, omit from new configs.
                continue

            if has_previous:
                file.write('\n\n')
            has_previous = True
            for line in option.doc:
                file.write(f'\t// {line}\n')

            # PROP types are "raw", so they don't have defaults.
            if option.kind is not Keyvalues and isinstance(option, OptWithDefault):
                if isinstance(default, bool):
                    default = '1' if default else '0'

                file.write(f'\t// - Default Value: "{default}"\n')

            match value:
                case None:
                    # Comment out the unset value.
                    file.write('\t// - Disabled by default, remove "//" to enable.\n')
                    file.write(f'\t// "{option.name}" ""\n')
                case Keyvalues():
                    value.name = option.name
                    value.serialise(file, start_indent='\t')
                case bool():
                    file.write(f'\t"{option.name}" "{'1' if value else '0'}"\n')
                case _:
                    file.write(f'\t"{option.name}" "{value}"\n')
        file.write('\t}\n')
