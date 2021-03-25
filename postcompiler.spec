"""Build the postcompiler script."""
import os
from pathlib import Path


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
        'statistics', 'string', 'struct', 'srctools',
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
