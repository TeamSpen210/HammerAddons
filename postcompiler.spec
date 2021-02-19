"""Build the postcompiler script."""
import os
from pathlib import Path


# Find the BSP transforms from HammerAddons.
try:
    transform_loc = Path(os.environ['BSP_TRANSFORMS']).resolve()
except KeyError:
    transform_loc = Path('../HammerAddons/transforms/').resolve()
if not transform_loc.exists():
    raise ValueError(
        f'Invalid BSP transforms location "{transform_loc}"!\n'
        'Clone TeamSpen210/HammerAddons, or set the '
        'environment variable BSP_TRANSFORMS to the location.'
    )

DATAS = [
    (str(file), str('transforms' / file.relative_to(transform_loc).parent))
    for file in transform_loc.rglob('*.py')
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
