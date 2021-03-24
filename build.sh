#!/bin/sh
mkdir -p build
cp -rf hammer build/hammer
cp -rf instances build/instances
find ./build/instances -iname "*.vmx" -delete # Yes, I know that we could use rsync with a ton of options to do this instead of using cp and then deleting unwanted files. This is FAR nicer imo.
python3 unify_fgd.py exp p2 srctools -o "build/portal2.fgd"
python3 unify_fgd.py exp p1 srctools -o "build/portal.fgd"
python3 unify_fgd.py exp hl2 srctools -o "build/hl2.fgd"
python3 unify_fgd.py exp ep1 ep2 srctools -o "build/episodic.fgd"
python3 unify_fgd.py exp gmod srctools -o "build/gmod.fgd"
python3 unify_fgd.py exp csgo srctools -o "build/csgo.fgd"
python3 unify_fgd.py exp tf2 srctools -o "build/tf2.fgd"
python3 unify_fgd.py exp asw srctools -o "build/asw.fgd"
python3 unify_fgd.py exp l4d srctools -o "build/l4d.fgd"
python3 unify_fgd.py exp l4d2 srctools -o "build/l4d2.fgd"
python3 unify_fgd.py exp infra srctools -o "build/infra.fgd"
python3 unify_fgd.py exp mesa srctools -o "build/blackmesa.fgd"
python3 unify_fgd.py exp p2ce srctools -o "build/p2ce.fgd"
