"""Build the postcompiler script."""
from pathlib import Path
import shutil
import importlib.metadata

from PyInstaller.utils.hooks import collect_submodules
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

src_version = importlib.metadata.version('srctools')

with open(Path(SPECPATH, 'src', 'hammeraddons', '_version.py'), 'w') as f:
    f.write(f'HADDONS_VER = {version!r}\nSRCTOOLS_VER = {src_version!r}\n')

DATAS = [
    (str(root / 'crowbar_command/Crowbar.exe'), '.'),
    (str(root / 'crowbar_command/FluentCommandLineParser.dll'), '.'),
]

a = Analysis(
    ['src/hammeraddons/postcompiler.py'],
    binaries=[],
    datas=DATAS,
    hiddenimports=[
        # Ensure these modules are available for plugins.
        'abc', 'array', 'base64', 'binascii', 'graphlib',
        'bisect', 'colorsys', 'collections', 'csv', 'datetime', 'contextlib',
        'decimal', 'difflib', 'enum', 'fractions', 'functools',
        'io', 'itertools', 'json', 'math', 'random', 're',
        'statistics', 'string', 'struct',
        *collect_submodules('srctools', filter=lambda name: 'scripts' not in name),
        *collect_submodules('attr'),
        *collect_submodules('attrs'),
        *collect_submodules('hammeraddons'),
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
    # Don't use bin/, in case someone puts this right in a game dir.
    contents_directory='binaries',
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

# Copy transforms to the same place as the EXE, not into the binaries subfolder.
app_folder = Path(coll.name)
for file in (root / 'transforms').rglob('*.py'):
    dest = app_folder / file.relative_to(root)
    print(file, '->', dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, dest)
