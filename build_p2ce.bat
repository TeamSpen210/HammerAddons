robocopy hammer build/hammer  /S /PURGE
robocopy instances build/instances /XF *.vmx /S /PURGE
python unify_fgd.py exp p2ce srctools -o "build/p2ce.fgd"
