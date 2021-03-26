// Custom model to work around the scale keyvalue making it massive.

$modelname "editor/env_dustpuff.mdl"

$body "Body" "env_dustpuff_ref.smd"

$staticprop

$cdmaterials "models/editor/ts/"

$sequence idle "env_dustpuff_ref"

// Thicken selection to grid axes.
$hbox 0 "static_prop" -8 -8 -8 8 8 8
