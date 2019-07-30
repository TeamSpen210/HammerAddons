"""Property-based configuration system.

A list of options are passed in, which parse each option to a basic type.
"""
import inspect
from enum import Enum, EnumMeta
from pathlib import Path
from typing import TypeVar, Union, Any, List, Type, Optional, Dict, IO, overload

from srctools import Vec, Property, parse_vec_str, conv_bool
from srctools.logger import get_logger

LOGGER = get_logger(__name__)


class TYPE(Enum):
    """The types arguments can have."""
    STR = str
    INT = int
    FLOAT = float
    BOOL = bool
    VEC = Vec
    RAW = Property  # This bypasses parsing, giving you the raw block.

    def convert(self, value: str) -> Any:
        """Convert a string to the desired argument type."""
        return self.value(value)


TYPE_NAMES = {
    TYPE.STR: 'Text',
    TYPE.INT: 'Whole Number',
    TYPE.FLOAT: 'Decimal Number',
    TYPE.BOOL: 'True/False',
    TYPE.VEC: 'Vector',
    TYPE.RAW: 'Property Block',
}

OptionT = TypeVar('OptionT', str, int, float, bool, Vec)
EnumT = TypeVar('EnumT', bound=Enum)


class Opt:
    """A type of option that can be chosen."""
    default: Union[None, str, int, float, bool, Vec, Property]

    def __init__(
        self,
        opt_id: str,
        default: Union[TYPE, OptionT, Property],
        doc: str,
        fallback: str=None,
    ) -> None:
        if isinstance(default, TYPE):
            self.type = default
            if default is TYPE.RAW:
                self.default = Property(opt_id, [])
            else:
                self.default = None
        else:
            self.type = TYPE(type(default))
            self.default = default
        self.id = opt_id.casefold()
        self.name = opt_id
        self.fallback = fallback
        # Remove indentation, and trailing carriage return
        self.doc = inspect.cleandoc(doc).rstrip().splitlines()
        if fallback is not None:
            self.doc.append(
                'If unset, the default is read from `{}`.'.format(default)
            )


class Config:
    """Allows parsing a set of Property option blocks."""
    def __init__(self, defaults: Union[List[Opt], 'Config']) -> None:
        if isinstance(defaults, Config):
            self.defaults = defaults.defaults  # type: List[Opt]
        else:
            self.defaults = defaults

        self.settings = {}  # type: Dict[str, Union[None, str, int, float, bool, Vec, Property]]
        self.path = None  # type: Optional[Path]

    def load(self, opt_blocks: Property) -> None:
        """Read settings from the given property block."""
        self.settings.clear()
        set_vals = {}
        for opt_block in opt_blocks:
            for prop in opt_block:
                set_vals[prop.name] = prop

        options = {opt.id: opt for opt in self.defaults}
        if len(options) != len(self.defaults):
            from collections import Counter
            # Find ids used more than once..
            raise Exception('Duplicate option(s)! ({})'.format(', '.join(
                k for k, v in
                Counter(opt.id for opt in self.defaults).items()
                if v > 1
            )))

        fallback_opts = []

        for opt in self.defaults:
            try:
                prop = set_vals.pop(opt.id)
            except KeyError:
                if opt.fallback is not None:
                    fallback_opts.append(opt)
                    assert opt.fallback in options, 'Invalid fallback in ' + opt.id
                else:
                    self.settings[opt.id] = opt.default
                continue
            if opt.type is TYPE.RAW:
                self.settings[opt.id] = prop.copy()
                continue

            # Non-RAW types cannot have a property block, only a value.
            if prop.has_children():
                raise ValueError(
                    'Cannot use property block for '
                    '"{}"'.format(opt.name)
                )

            if opt.type is TYPE.VEC:
                # Pass nones so we can check if it failed..
                parsed_vals = parse_vec_str(prop.value, x=None)
                if parsed_vals[0] is None:
                    self.settings[opt.id] = opt.default
                else:
                    self.settings[opt.id] = Vec(*parsed_vals)
            elif opt.type is TYPE.BOOL:
                self.settings[opt.id] = conv_bool(prop.value, opt.default)
            else:  # int, float, str - no special handling...
                try:
                    self.settings[opt.id] = opt.type.convert(prop.value)
                except (ValueError, TypeError):
                    self.settings[opt.id] = opt.default

        for opt in fallback_opts:
            assert opt.fallback is not None
            try:
                self.settings[opt.id] = self.settings[opt.fallback]
            except KeyError:
                raise Exception('Bad fallback for "{}"!'.format(opt.id))
            # Check they have the same type.
            if opt.type is not options[opt.fallback].type:
                raise ValueError(
                    '"{}" cannot fall back to "{}" - different '
                    'type!'.format(opt.id, opt.fallback)
                )

        if set_vals:
            LOGGER.warning('Extra config options: {}', set_vals)

    def set_opt(self, opt_name: str, value: str) -> None:
        """Set an option to a specific value."""
        folded_name = opt_name.casefold()
        for opt in self.defaults:
            if folded_name == opt.id:
                break
        else:
            LOGGER.warning('Invalid option name "{}"!', opt_name)
            return

        if opt.type is TYPE.RAW:
            if not isinstance(value, Property):
                raise ValueError(
                    'The value must be a Property '
                    'for property blocks!'
                )
            self.settings[opt.id] = value
        elif opt.type is TYPE.VEC:
            # Pass nones so we can check if it failed..
            parsed_vals = parse_vec_str(value, x=None)
            if parsed_vals[0] is None:
                return
            self.settings[opt.id] = Vec(*parsed_vals)
        elif opt.type is TYPE.BOOL:
            self.settings[opt.id] = conv_bool(value, self.settings[opt.id])
        else:  # int, float, str - no special handling...
            try:
                self.settings[opt.id] = opt.type.convert(value)
            except (ValueError, TypeError):
                pass

    @overload
    def get(self, expected_type: Type[Property], name: str) -> Property: ...

    @overload
    def get(self, expected_type: Type[EnumT], name: str) -> EnumT: ...

    @overload
    def get(self, expected_type: Type[OptionT], name: str) -> Optional[OptionT]: ...

    def get(self, expected_type: type, name: str) -> Any:
        """Get the given option.
        expected_type should be the class of the value that's expected.
        The value can be None if unset, except for Property types (which
        will always have an empty block).

        If expected_type is an Enum, this will be used to convert the output.
        If it fails, a warning is produced and the first value in the enum is
        returned.
        """
        try:
            val = self.settings[name.casefold()]
        except KeyError:
            raise TypeError(
                'Option "{}" does not exist!'.format(name)
            ) from None

        if val is None:
            if expected_type is Property:
                return Property(name, [])
            else:
                return None

        if issubclass(expected_type, Enum):
            enum_type = expected_type  # type: Optional[Type[Enum]]
            expected_type = str
        else:
            enum_type = None

        # Don't allow subclasses (bool/int)
        if type(val) is not expected_type:
            raise ValueError('Option "{}" is {} (code expected {})'.format(
                name,
                type(val),
                expected_type,
            ))

        if enum_type is not None:
            try:
                return enum_type(val)  # type: ignore
            except ValueError:
                LOGGER.warning(
                    'Option "{}" is not a valid value. '
                    'Allowed values are:\n{}',
                    name,
                    '\n'.join([mem.value for mem in enum_type])
                )
                return next(iter(enum_type))  # type: ignore

        # Vec is mutable, don't allow modifying the original.
        if expected_type is Vec or expected_type is Property:
            assert isinstance(val, Vec) or isinstance(val, Property)
            return val.copy()
        else:
            assert isinstance(val, expected_type)
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

            default = option.default

            # PROP types are "raw", so they don't have defaults.
            if option.type is not TYPE.RAW and default is not None:
                if isinstance(default, bool):
                    default = '1' if default else '0'

                file.write('\t// Default Value: "{}"\n'.format(default))

            try:
                value = self.settings[option.id]
            except KeyError:
                value = default

            if isinstance(value, bool):
                value = '1' if value else '0'

            if value is None:
                # Comment out the unset value.
                file.write('\t// "{}" ""\n'.format(option.name))
            elif isinstance(value, Property):
                value.name = option.name
                for line in value.export():
                    file.write('\t' + line)
            else:
                file.write('\t"{}" "{}"\n'.format(option.name, value))
        file.write('\t}\n')
