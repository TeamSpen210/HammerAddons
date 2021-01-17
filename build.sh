#!/bin/sh
mkdir -p build
cp -rf hammer build/hammer
cp -rf instances build/instances
cp -rf transforms build/postcompiler/transforms
find ./build/instances -iname "*.vmx" -delete # Yes, I know that we could use rsync with a ton of options to do this instead of using cp and then deleting unwanted files. This is FAR nicer imo.
python3 unify_fgd.py exp p2ce srctools -o "build/p2ce.fgd"
python3 unify_fgd.py exp momentum srctools -o "build/momentum.fgd"
