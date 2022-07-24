// Modified version with no anims, fixed physics.

$ModelName "editor/item_suit_charger_mesa.mdl"

$BodyGroup "body"
{
	studio "hev_charger_reference.smd"
}

$SurfaceProp "computer"

$Contents "solid"

$IllumPosition 4.066 0 0

$CDMaterials "models\props_blackmesa\"

$TextureGroup "skinfamilies"
{
	{ "bms_hev_charger_clean_metal"     "bms_hev_charger1_screen"  }
	// { "bms_HEV_charger_clean_metal_off" "bms_HEV_charger1_screen2" }
	{ "bms_hev_charger1_metal"     "bms_hev_charger1_screen"  }
	// { "bms_HEV_charger1_metal_off" "bms_HEV_charger1_screen2" }
}

$Attachment "dlight" "Charger_Base" -9 5 -4 rotate 90 0 0

$CBox 0 0 0 0 0 0

$BBox -17.84 -0.2 -22.727 12 8.331 18

$SectionFrames 30 120

$Sequence "idle" {
	"hev_charger_reference.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}

$CollisionModel "hev_charger_physics.smd"
{
	$mass 1
	$inertia 1
	$damping 0
	$rotdamping 0
	$rootbone " "
}
