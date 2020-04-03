// Make this model have no animation, so it doesn't loop in Hammer.

$ModelName "editor/item_health_charger_hl2.mdl"

$staticprop

$BodyGroup "Body"
{
	studio "health_charger001_reference.smd"
}

$LOD 10
{
	replacemodel "health_charger001_reference.smd" "health_charger001_reference_lod1.smd"
	NoFacial
}
$LOD 20
{
	replacemodel "health_charger001_reference.smd" "health_charger001_reference_lod2.smd"
	NoFacial
}
$LOD 30
{
	replacemodel "health_charger001_reference.smd" "health_charger001_reference_lod3.smd"
	NoFacial
}

$SurfaceProp "metal"

$Contents "solid"

$IllumPosition 0.779 -3.793 -0.016

$MostlyOpaque

$CDMaterials "models\props_combine\"

$CBox 0 0 0 0 0 0

$BBox -0.281 -10.153 -20.244 7.733 11.834 19.951

$Sequence "idle" {
	"idle.smd"
}

$CollisionModel "health_charger001_physics.smd"
{
	$mass 30
	$inertia 1
	$damping 0
	$rotdamping 0
	$rootbone " "

}
