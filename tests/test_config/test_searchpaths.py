"""Test searchpath handling."""
import pytest

# noinspection PyProtectedMember
from hammeraddons.config import SearchpathEntry
from srctools import Keyvalues
from srctools.dmx import Element


@pytest.mark.parametrize('filename, kind', [
    ('foLDer/', 'folder'),
    ('foLD*er/', 'folder'),
    ('pak01_DIR.vPk', 'vpk')
])
def test_parse_searchpath_kv_str(filename: str, kind: SearchpathEntry.Kind) -> None:
    """Test parsing one-line keyvalue definitions."""
    assert SearchpathEntry.parse_kv(
        Keyvalues('path', filename), False,
    ) == SearchpathEntry(filename, kind, mode='append', pack=None)

    assert SearchpathEntry.parse_kv(
        Keyvalues('preFIx', filename), False,
    ) == SearchpathEntry(filename, kind, mode='prepend', pack=None)

    assert SearchpathEntry.parse_kv(
        Keyvalues('prIOrity', filename), False,
    ) == SearchpathEntry(filename, kind, mode='prepend', pack=None)

    assert SearchpathEntry.parse_kv(
        Keyvalues('nopack', filename), False,
    ) == SearchpathEntry(filename, kind, mode='append', pack=False)
    assert SearchpathEntry.parse_kv(
        Keyvalues('nopack', filename), True,
    ) == SearchpathEntry(filename, kind, mode='optional', pack=False)

    assert SearchpathEntry.parse_kv(
        Keyvalues('pack', filename), False,
    ) == SearchpathEntry(filename, kind, mode='append', pack=True)
    assert SearchpathEntry.parse_kv(
        Keyvalues('pack', filename), True,
    ) == SearchpathEntry(filename, kind, mode='optional', pack=True)


@pytest.mark.parametrize('filename, kind', [
    ('foLDer/', 'folder'),
    ('foLD*er/', 'folder'),
    ('pak01_DIR.vPk', 'vpk')
])
def test_parse_searchpath_kv_block(filename: str, kind: SearchpathEntry.Kind) -> None:
    """Test parsing block keyvalue definitions."""
    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
    ]), False) == SearchpathEntry(filename, kind, mode='append', pack=None)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename)
    ]), True) == SearchpathEntry(filename, kind, mode='optional', pack=None)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('priority', '0'),
    ]), False) == SearchpathEntry(filename, kind, mode='append', pack=None)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('priority', '0'),
    ]), True) == SearchpathEntry(filename, kind, mode='append', pack=None)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('priority', '1'),
    ]), True) == SearchpathEntry(filename, kind, mode='prepend', pack=None)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('pack', '0'),
    ]), True) == SearchpathEntry(filename, kind, mode='optional', pack=False)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('pack', '1'),
    ]), True) == SearchpathEntry(filename, kind, mode='optional', pack=True)

    assert SearchpathEntry.parse_kv(Keyvalues('', [
        Keyvalues('path', filename),
        Keyvalues('pack', '1'),
        Keyvalues('priority', '1')
    ]), True) == SearchpathEntry(filename, kind, mode='prepend', pack=True)


@pytest.mark.parametrize('filename, dmxtype, kind', [
    ('foLDer/', 'SearchFolder', 'folder'),
    ('foLD*er/', 'SearchFolder', 'folder'),
    ('pak01_DIR.vPk', 'SearchVPK', 'vpk')
])
def test_parse_searchpath_dmx(filename: str, dmxtype: str, kind: SearchpathEntry.Kind) -> None:
    """Test parsing DMX definitions."""
    elem = Element('Entry', dmxtype)
    elem['path'] = filename
    assert SearchpathEntry.parse_dmx(
        elem, False,
    ) == SearchpathEntry(filename, kind, mode='append', pack=None)
    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='optional', pack=None)

    elem = Element('Entry', dmxtype)
    elem['paTH'] = filename
    elem['priORity'] = False
    assert SearchpathEntry.parse_dmx(
        elem, False
    ) == SearchpathEntry(filename, kind, mode='append', pack=None)

    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='append', pack=None)

    elem = Element('Entry', dmxtype)
    elem['patH'] = filename
    elem['prioRIty'] = True
    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='prepend', pack=None)

    elem = Element('Entry', dmxtype)
    elem['path'] = filename
    elem['paCK'] = False
    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='optional', pack=False)

    elem = Element('Entry', dmxtype)
    elem['Path'] = filename
    elem['pACk'] = True
    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='optional', pack=True)

    elem = Element('Entry', dmxtype)
    elem['path'] = filename
    elem['prIOrity'] = True
    elem['pack'] = True
    assert SearchpathEntry.parse_dmx(
        elem, True
    ) == SearchpathEntry(filename, kind, mode='prepend', pack=True)
