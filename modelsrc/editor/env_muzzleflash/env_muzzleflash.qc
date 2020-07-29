$ModelName "editor/env_muzzleflash.mdl"

$BodyGroup "default"
{
	studio "muzzleflash_model.smd"
}

$staticprop

$SurfaceProp "default"

$Contents "solid"

$IllumPosition 0 0 0

$CDMaterials "models/editor/ts"
$CDMaterials "editor"

$CBox 0 0 0 0 0 0

$BBox -8 -8 -8 8 8 8
$hbox 0 "static_prop" -8 -8 -12  2 8 8

$Sequence "BindPose" muzzleflash_model.smd

