"""Build the postcompiler script."""
import os
from pathlib import Path

import versioningit


# Find the BSP transforms from HammerAddons.
try:
    hammer_addons = Path(os.environ['HAMMER_ADDONS']).resolve()
except KeyError:
    hammer_addons = Path('../HammerAddons/').resolve()
if not (hammer_addons / 'transforms').exists():
    raise ValueError(
        f'Invalid BSP transforms location "{hammer_addons}/transforms/"!\n'
        'Clone TeamSpen210/HammerAddons, or set the '
        'environment variable HAMMER_ADDONS to the location.'
    )

version = versioningit.get_version(hammer_addons, {
    'vcs': {'method': 'git'},
    'default-version': '(dev)',
    'format': {
        'distance': '{version}.dev_{distance}+{rev}',
        'dirty': '{version}+dirty_{build_date:%Y%m%d}',
        'distance-dirty': '{version}.dev_{distance}+{rev}.dirty_{build_date:%Y%m%d}',
    },
})

with open(Path(SPECPATH, 'srctools', 'compiler', '_version.py'), 'w') as f:
    f.write(f'__version__ = {version!r}\n')

DATAS = [
    (str(file), str(file.relative_to(hammer_addons).parent))
    for file in (hammer_addons / 'transforms').rglob('*.py')
] + [
    (str(hammer_addons / 'crowbar_command/Crowbar.exe'), '.'),
    (str(hammer_addons / 'crowbar_command/FluentCommandLineParser.dll'), '.'),
]
print(DATAS)

a = Analysis(
    ['srctools/scripts/postcompiler.py'],
    binaries=[],
    datas=DATAS,
    hiddenimports=[
        # Ensure these modules are available for plugins.
        'abc', 'array', 'base64', 'binascii', 'binhex',
        'bisect', 'colorsys', 'collections', 'csv', 'datetime',
        'decimal', 'difflib', 'enum', 'fractions', 'functools',
        'io', 'itertools', 'json', 'math', 'random', 're',
        'statistics', 'string', 'struct',
        'srctools', 'attrs',
    ],
    excludes=['srctools.test'],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='postcompiler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon="postcompiler.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='postcompiler'
)
