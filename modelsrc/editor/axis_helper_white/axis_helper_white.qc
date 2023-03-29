// A version of the axis helper, designed to tint along with sprites etc.

$modelname "editor/axis_helper_white.mdl"

$bodygroup "Body"
{
	studio "axis_helper.smd"
}


$surfaceprop "default"

$contents "solid"

$illumposition 0 0 0

$cdmaterials "models\editor\"

$cbox 0 0 0 0 0 0

$bbox -5.5 -5.5 -5.5 5.5 5.5 5.5

$sequence "idle" {
	"idle.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}
