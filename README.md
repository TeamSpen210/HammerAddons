<div align="center">
    <img src="logo/icon_256.png" alt="Hammer Addons" height="200" />
	<br>
	<br>
	<p> <b>Teamspen's Hammer Addons </b></p>
</div>

<hr>

[releases]: https://github.com/TeamSpen210/HammerAddons/releases
[installationwiki]: https://github.com/TeamSpen210/HammerAddons/wiki/Installation
[skotty]: http://forums.thinking.withportals.com/downloads.php?view=detail&df_id=507
[tf2]: http://forums.tf2maps.net/showthread.php?t=4674
[ts2do]: http://halflife2.filefront.com/file/HalfLife_2_Upgraded_Base_FGDs;48139
[zps]: http://www.necrotalesgames.com/tools/index.php


## Features

* Auto-packing - filtered based on search paths in gameinfo, and based on a FGD database + `comp_pack` entities.
* Static prop combining - merges together adjacent props to allow them to be efficently drawn in batches.
* A unified FGD database allowing keyvalues to be shared among games, and accurately defining when features were added and removed.
* Many many upgrades to entity options and layouts.
* New sprites for almost all entities, both custom made and from a number of [other sources](#development).
* Adds lots more AutoVisgroups for easily hiding entities.
* Several `comp_` entities with additional features. These are mainly intended for use in instances, allowing modifying entities outside of the instance to conform or doing normally impossible things like positioning things in the void.
	To use, decompile props and configure the folder & studioMDL's path then place `comp_propcombine_set` entities.
* For games supporing VScript:
	* In any `RunScriptCode` input, backticks can be used for string literals, instead of the disallowed `"` character. 
	* In addition to the normal `Entity Scripts` section, a new `Init Code` field can be used to write code that's packed and added to those scripts. Useful for setting configuration options etc. Backticks can be used here too.

## Installation

* Download the latest release from the [releases tab][releases].
* Follow [this guide][installationwiki].
* If using BEEMOD2.4, change Hammer -> Options -> Build Programs to use `vrad_original.exe`.

## Using Vactubes (Portal 2 only)

This implements a dynamic vactube system in a similar way to Valve's system, including randomised objects, complex junctions and dropper support.

* To use, place and configure `comp_vactube_object` entities to specify which items can appear in tubes.
* To build paths a comp_vactube_start entity at the beginning of the track, and a `comp_vactube_end` at the end. 
* Then at each corner/junction place a `comp_vactube_junction` ent, picking the appropriate type. These all need to be rotated appropriately so the arrows point in the correct direction to be matched up by the compiler. 
* To split a path into multiple tubes, you'll need to use one of the "splitter" junction types. 
* To join multiple back into a single pipe, simply overlap two junctions such that their outputs both point down the same route. 
* For droppers, simply place the supplied `instances/cubedropper/dropper_vactube.vmf` instance, and run a path up to the vactube end entity in the top. Place a `prop_weighted_cube` inside the dropper to specify which cube type it will spawn. The specific route leading to the dropper will be detected and only replacement cubes will be sent this way. You'll want to add a splitter just before the dropper, so the tube can have decorative items flowing through it constantly. 
* To place the vactube scanner TVs, simply add a "straight"-type junction inside the model, then place the `prop_dynamic`s for the screen and optionally the spinner. The screen will need the supplied `_new` model, so both orientations have all the skins. They will automatically be detected and flash on when objects pass.
* To avoid visual collisions, you may want to turn off the automatic spawning on one or more spawn points, then use the outputs on a junction to manually spawn objects in sync with another path.

## Development

* Mapbase's FGDs have been imported as a submodule.
* Some entity sprites are taken from: 
  * [The TF2 Ultimate Mapping Resource Pack][tf2]
  * [ZPS: Supplemental Hammer Icons][zps]
  * [ts2do's HL FGDs][ts2do]
