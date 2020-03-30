$modelname "editor/prop_paint_bomb.mdl"

$body "Body" "prop_paint_bomb_ref.smd"

$staticprop

$cdmaterials "models/editor/ts/"

$sequence idle "prop_paint_bomb_ref"

$hbox 0 "static_prop" -28 -28 -28 28 28 28

$texturegroup skinfamilies {
	{ "blob_surface_bounce" }
	{ "blob_surface_reflect" }
	{ "blob_surface_speed" }
	{ "blob_surface_portal" }
	{ "blob_surface_erase" }
	{ "blob_surface_stick" }
}
