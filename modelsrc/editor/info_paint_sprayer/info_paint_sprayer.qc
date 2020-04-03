$modelname "editor/info_paint_sprayer.mdl"

$body "Body" "info_paint_sprayer_ref.smd"

$staticprop

$cdmaterials "models/editor/"
$cdmaterials "models/editor/ts/"

$sequence idle "info_paint_sprayer_ref"

// Thicken selection to grid axes.
$hbox 0 "static_prop" -10 -10 -10 10 10 10

$texturegroup skinfamilies {
	{ "blob_surface_bounce" }
	{ "blob_surface_reflect" }
	{ "blob_surface_speed" }
	{ "blob_surface_portal" }
	{ "blob_surface_erase" }
	{ "blob_surface_stick" }
}
