"""Build the postcompiler script."""


a = Analysis(
    ['srctools/scripts/postcompiler.py'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Ensure these modules are available for plugins.
        'abc', 'array', 'base64', 'binascii', 'binhex',
        'bisect', 'colorsys', 'collections', 'csv', 'datetime',
        'decimal', 'difflib', 'enum', 'fractions', 'functools',
        'io', 'itertools', 'json', 'math', 'random', 're',
        'statistics', 'string', 'struct',
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
