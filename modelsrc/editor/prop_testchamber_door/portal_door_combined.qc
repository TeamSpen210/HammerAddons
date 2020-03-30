// Created by Crowbar 0.55

$ModelName "editor/prop_testchamber_door.mdl"

$BodyGroup "default"
{
	studio "portal_door_combined_model.smd"
}

$staticprop

$cdmaterials "models/props/"

$sequence idle "portal_door_combined_model"

// Thicken selection to grid axes.
$hbox 0 "static_prop" -20.00 -112 0   20 112 116

// For visuals...
$collisionmodel "portal_door_combined_physics.smd"
	{
	$concave
	$mass 7639.53
	$inertia 1
	$damping 0.35
	$rotdamping 0
	}
