// Modified version with no anims, fixed physics.

$ModelName "editor/item_health_charger_mesa.mdl"

$BodyGroup "body"
{
	studio "health_charger.smd"
}

$SurfaceProp "computer"

$Contents "solid"

$IllumPosition 2.5 0 0

$CDMaterials "models\props_blackmesa\"

$TextureGroup "skinfamilies"
{
	{ "bms_health_charger_clean_bulbs"     "bms_health_charger_clean_metalblue"     "bms_health_charger_clean_metalred"     "bms_health_charger_screen1"    }
	// { "bms_health_charger_clean_bulbs_off" "bms_health_charger_clean_metalblue_off" "bms_health_charger_clean_metalred_off" "bms_health_charger_screen2"    }
	// { "bms_health_charger_clean_bulbs_off" "bms_health_charger_clean_metalblue_off" "bms_health_charger_clean_metalred_off" "bms_health_charger_screen_off" }

	{ "bms_health_charger1_bulbs"     "bms_health_charger1_metalblue"     "bms_health_charger1_metalred"     "bms_health_charger_screen1"    }
	// { "bms_health_charger1_bulbs_off" "bms_health_charger1_metalblue_off" "bms_health_charger1_metalred_off" "bms_health_charger_screen2"    }
	// { "bms_health_charger1_bulbs_off" "bms_health_charger1_metalblue_off" "bms_health_charger1_metalred_off" "bms_health_charger_screen_off" }
}

$Attachment "dlight" "ChargerBase001" -8 3 -5 rotate 90 0 0

$CBox 0 0 0 0 0 0

$BBox -12 0 -18 12 5 18

$Sequence "idle" {
	"health_charger.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}

$CollisionModel "health_charger_physics.smd"
{
	$mass 1
	$inertia 1
	$damping 0
	$rotdamping 0
}
