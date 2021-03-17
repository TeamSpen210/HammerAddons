robocopy hammer build/hammer  /S /PURGE
robocopy instances build/instances /XF *.vmx /S /PURGE
python unify_fgd.py exp p2 srctools -o "build/portal2.fgd"
python unify_fgd.py exp p1 srctools -o "build/portal.fgd"
python unify_fgd.py exp hl2 srctools -o "build/hl2.fgd"
python unify_fgd.py exp ep1 ep2 srctools -o "build/episodic.fgd"
python unify_fgd.py exp gmod srctools -o "build/gmod.fgd"
python unify_fgd.py exp csgo srctools -o "build/csgo.fgd"
python unify_fgd.py exp tf2 srctools -o "build/tf2.fgd"
python unify_fgd.py exp l4d srctools -o "build/l4d.fgd"
python unify_fgd.py exp l4d2 srctools -o "build/l4d2.fgd"
python unify_fgd.py exp infra srctools -o "build/infra.fgd"
python unify_fgd.py exp mesa srctools -o "build/blackmesa.fgd"
python unify_fgd.py exp p2ce srctools -o "build/p2ce.fgd"
