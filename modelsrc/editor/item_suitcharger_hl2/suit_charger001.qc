// Make this model have no animation, so it doesn't loop in Hammer.

$ModelName "editor/item_suit_charger_hl2.mdl"

$staticprop

$BodyGroup "Body"
{
	studio "suit_charger001_reference.smd"
}

$LOD 10
{
	replacemodel "suit_charger001_reference.smd" "suit_charger001_reference_lod1.smd"
	NoFacial
}
$LOD 20
{
	replacemodel "suit_charger001_reference.smd" "suit_charger001_reference_lod2.smd"
	NoFacial
}
$LOD 40
{
	replacemodel "suit_charger001_reference.smd" "suit_charger001_reference_lod3.smd"
	NoFacial
}

$SurfaceProp "metal"

$Contents "solid"

$CDMaterials "models\props_combine\"


$Sequence "idle" {
	"idle.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}


$CollisionModel "suit_charger001_physics.smd"
{
	$mass 220
	$concave
	$maxconvexpieces 2
}
