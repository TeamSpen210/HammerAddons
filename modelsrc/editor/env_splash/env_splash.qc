$modelname "editor/env_splash.mdl"

$body "Body" "env_splash_ref.smd"

$staticprop

$cdmaterials "models/editor/ts/"

$sequence idle "env_splash_ref"

// Thicken selection to grid axes.
$hbox 0 "static_prop" -8 -8 -8 8 8 8
