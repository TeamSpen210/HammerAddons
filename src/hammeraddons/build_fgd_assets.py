"""Creates the subset of assets needed for an input FGD to work.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import os

from srctools.fgd import (
    FGD, AutoVisgroup, EntAttribute, EntityDef, EntityTypes, Helper, HelperExtAppliesTo,
    HelperTypes, KVDef, Snippet, ValueTypes, match_tags, validate_tags,
)
from srctools.filesys import File, RawFileSystem
from srctools.packlist import PackList


def action_build(input_path: Path, output_path: Path, asset_path: Path) -> None:

    # Import the FGD
    fgd_fsys = RawFileSystem(str(input_path.parent))
    asset_fsys = RawFileSystem(str(asset_path))
    fgd = FGD()
    fgd.parse_file(fgd_fsys, fgd_fsys[str(input_path)], eval_bases=False, eval_extensions=False, encoding='iso-8859-1')
    pack = PackList(asset_fsys)

    # Iterate over all entries, build a list of assets
    for classname in fgd.entities:
        ent = fgd.entities[classname]
        for helper in ent.helpers:
            for resource in helper.get_resources(ent):
                pack.pack_file(resource)

    # Evaluate everything we needed for these helper's resources
    pack.eval_dependencies()

    # Output all the files we can to our new output
    for file in pack._files.values():
        try:
            sys_file = asset_fsys[file.filename]
        except FileNotFoundError:
            print(f'WARNING: "{file.filename}" not packed!')
            continue

        print(file.filename)
        with sys_file.open_bin() as f:
            data = f.read()
            new_path = Path(os.path.join(str(output_path), file.filename))
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with open(new_path, "wb") as out:
                out.write(data)


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Build a set of assets from an input FGD.",
    )

    parser.add_argument(
        "-i", "--input",
        help="The FGD to read from."
    )
    parser.add_argument(
        "-o", "--output",
        help="Output folder",
    )
    parser.add_argument(
        "-a", "--assets",
        help="Assets folder",
    )
    result = parser.parse_args(args)

    if result.input is None or result.output is None or result.assets is None:
        parser.print_help()
        return

    input_path = Path(result.input).resolve()
    output_path = Path(result.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    asset_path = Path(result.assets).resolve()

    action_build(input_path, output_path, asset_path)

if __name__ == '__main__':
    main(sys.argv[1:])
