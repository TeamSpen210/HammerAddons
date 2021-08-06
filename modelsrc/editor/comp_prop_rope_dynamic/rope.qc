$ModelName "editor/comp_prop_rope_dynamic.mdl"

$BodyGroup "default"
{
	studio "rope_ref.smd"
}

$staticprop

$CDMaterials "models/editor/ts"

$BBox -16 -4 -8 16 4 2
$hbox 0 "static_prop" -16 -4 -8 16 4 2

$Sequence "BindPose" rope_ref.smd

