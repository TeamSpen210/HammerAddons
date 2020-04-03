$staticprop
$modelname "editor/prop_portal"
$scale "1.000000"
$body "Body" "prop_portal_ref"
$cdmaterials "models/editor/ts/"
$sequence idle "prop_portal_ref"
$surfaceprop "no_decal"
$texturegroup skinfamilies {
	{ "portal_blue" }
	{ "portal_oran" }
	{ "portal_grey" }
}

// Thicken model, since it's 1-wide normally
$hbox 0 "static_prop" -4 -32 -56  4 32 56
