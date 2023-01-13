"""Build the postcompiler script."""
from pathlib import Path

import versioningit

# PyInstaller-injected.
SPECPATH: str
workpath: str

root = Path(SPECPATH)  # noqa

version = versioningit.get_version(SPECPATH, {
    'vcs': {'method': 'git'},
    'default-version': '(dev)',
    'format': {
        'distance': '{version}.dev_{distance}+{rev}',
        'dirty': '{version}+dirty_{build_date:%Y%m%d}',
        'distance-dirty': '{version}.dev_{distance}+{rev}.dirty_{build_date:%Y%m%d}',
    },
})

with open(Path(SPECPATH, 'src', 'hammeraddons', '_version.py'), 'w') as f:
    f.write(f'__version__ = {version!r}\n')

DATAS = [
    (str(file), str(file.relative_to(root).parent))
    for file in (root / 'transforms').rglob('*.py')
] + [
    (str(root / 'crowbar_command/Crowbar.exe'), '.'),
    (str(root / 'crowbar_command/FluentCommandLineParser.dll'), '.'),
]
for src, dest in DATAS:
    print(src, '->', dest)

a = Analysis(
    ['src/hammeraddons/postcompiler.py'],
    binaries=[],
    datas=DATAS,
    hiddenimports=[
        # Ensure these modules are available for plugins.
        'abc', 'array', 'base64', 'binascii', 'binhex', 'graphlib',
        'bisect', 'colorsys', 'collections', 'csv', 'datetime', 'contextlib',
        'decimal', 'difflib', 'enum', 'fractions', 'functools',
        'io', 'itertools', 'json', 'math', 'random', 're',
        'statistics', 'string', 'struct',
        'srctools', 'attr', 'attrs',
    ],
    excludes=[
        'IPython',  # Via trio
    ],
    noarchive=False,
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
