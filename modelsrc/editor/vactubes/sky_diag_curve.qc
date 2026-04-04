$modelname "editor/vactubes/diag_curve_sky.mdl"
$scale 0.0625 // 1/16

$staticprop

$model "studio" "diag_curve.smd"

$surfaceprop "default"
$contents "solid"

$cdmaterials "models/editor/"

$texturegroup "skinfamilies" {
	{ "vactube_arrow" }
	{ "vactube_arrow_rev" }
}

$sequence "idle" "diag_curve.smd"
