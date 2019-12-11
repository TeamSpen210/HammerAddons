"""Build the postcompiler script."""
from PyInstaller.utils.hooks import get_module_file_attribute
import os

srctools_hooks = os.path.dirname(get_module_file_attribute('srctools.pyinstaller_hook'))

a = Analysis(
    ['srctools/scripts/postcompiler.py'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[
     srctools_hooks,
    ],
    runtime_hooks=[],
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
