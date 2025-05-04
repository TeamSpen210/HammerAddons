<div style="text-align: center">
    <img src="logo/icon_256.png" alt="Hammer Addons" width=256 height=256 />
	<br>
	<br>
	<p> <b>Teamspen's Hammer Addons </b></p>
</div>

<hr>

## Features

* A "postcompiler" application which processes BSPs, allowing most of the new features.
* Auto-packing - Automatically packs non-stock game files into the bsp. Filtered based on search paths in the included custom gameinfo and FGD database. Assets can also be packed manually with `comp_pack` entities.
* Static prop combining - merges together adjacent props to allow them to be efficently drawn in batches. To use, specify studioMDL's path then place `comp_propcombine_volume` or `comp_propcombine_set` entities.
* A [unified FGD database][unifiedfgd], allowing keyvalues to be shared among games, and accurately defining when features were added and removed.
* Many more entity options, and an improved editor layout.
* New sprites for almost all entities, both custom made and from a number of [other sources](#development).
* Adds lots more AutoVisgroups for easily hiding entities.
* Improvements for games supporting VScript:
	* In any `RunScriptCode` input, backticks can be used for string literals, instead of the disallowed `"` character. 
	* In addition to the normal `Entity Scripts` section, a new `Init Code` field can be used to write code that's packed and added to those scripts. Useful for setting configuration options etc. Backticks can be used here too.
* New `comp_` entities. These are mainly intended for use in instances, allowing modifying entities outside of the instance to conform or doing normally impossible things like positioning things in the void.
Below are short explanations, see the "Help" display on the entity properties in Hammer for detailed functionality:

| Entity                                             | Description                                                                                                                                                                  |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `comp_adv_output`                                  | Adds a single output to another entity, while allowing instance renaming or `$fixups` to apply to only certain parts.                                                        |
| `comp_case`                                        | Version of `logic_case` which is optimised away by the compiler.                                                                                                             |
| `comp_choreo_sceneset`                             | Chains a set of choreographed scenes together.                                                                                                                               |
| `comp_entity_finder`                               | Finds the closest entity of a given type, then applies various transformations. Outputs from this entity will be moved to the found entity.                                  |
| `comp_entity_mover`                                | Shift an entity by a given amount. This is useful to place entities into the void, for example.                                                                              |
| `comp_flicker`                                     | Fires on/off and skin inputs repeatedly to simulate a flicker-on or off effect.                                                                                              |
| `comp_kv_setter`                                   | Sets a keyvalue on an entity to a new value. This is useful to compute spawnflags, or to adjust keyvalues when the target entity's options can't be set to a fixup variable. |
| `comp_numeric_transition`                          | When triggered, animates a keyvalue/input over time with various options.                                                                                                    |
| `comp_pack`                                        | Explicitly identify resources to pack into the map, in addition to automatic detection.                                                                                      |
| `comp_pack_rename`                                 | Pack a file into the BSP, under a different name than it starts with.                                                                                                        |
| `comp_pack_replace_soundscript`                    | Replace a soundscript with a different one.                                                                                                                                  |
| `comp_piston_platform`                             | Generates the appropriate logic to sequence a Portal-style piston platform.                                                                                                  |
| `comp_player_input_helper`                         | Fake entity that allows Hammer to autocomplete inputs fired at the special `!player` name (and other similar ones).                                                          |
| `comp_precache_model`                              | Force a specific model to load, for runtime switching. Duplicates will be removed.                                                                                           |
| `comp_precache_sound`                              | Force a specific sound to load, for runtime switching. Duplicates will be removed. More keyvalues can be added.                                                              |
| `comp_prop_cable`/`comp_prop_rope`                 | Generates 3D cables using a static prop.                                                                                                                                     |
| `comp_prop_cable_dynamic`/`comp_prop_rope_dynamic` | Modifies the above to generate a dynamic prop, instead.                                                                                                                      |
| `comp_prop_rope_bunting`                           | Adds additional props or geometry along the 3D cables.                                                                                                                       |
| `comp_propcombine_set`/`comp_propcombine_volume`   | Specifies a group of props that will be combined together, so they more efficiently render.                                                                                  |
| `comp_relay`                                       | Simplified version of `logic_relay` which is able to be optimised away by the compiler.                                                                                      |
| `comp_scriptvar_setter`                            | Assigns data or a group of data to a variable in an entity's VScript scope on spawn.                                                                                         |
| `comp_sequential_call`                             | Finds a sequence of entities (by distance or numeric suffix), then fires inputs delayed in order.                                                                            |
| `comp_vactube_end`                                 | Marks the end point of a vactube. Objects reaching here will be cleaned up.                                                                                                  |
| `comp_vactube_junction`                            | Marks a junction in a vactube, where they're forced to change direction. Scanner models near straight nodes will be detected automatically.                                  |
| `comp_vactube_object`                              | Registers objects that can appear in the tubing.                                                                                                                             |
| `comp_vactube_sensor`                              | Triggers outputs when a vactube object passes close by.                                                                                                                      |
| `comp_vactube_spline`                              | Generates a dynamic vactube model following a set of points.                                                                                                                 |
| `comp_vactube_start`                               | Marks the start point of a vactube. This is where they spawn.                                                                                                                |


## Installation

* Follow [this guide][installationwiki].
* If using BEEMOD2.4, change Hammer -> Options -> Build Programs to use `vrad_original.exe`.

## Credits

* Mapbase's FGDs have been imported as a submodule.
* Some entity sprites are taken from: 
  * [The TF2 Ultimate Mapping Resource Pack][tf2]
  * [ZPS: Supplemental Hammer Icons][zps]
  * [ts2do's HL FGDs][ts2do]
* Parts of [Allison's Portal 1 FGD edits][p1fgd] have been integrated into the fgds.

## Development

### Installing dependencies
The code requires Python 3.13, and is mainly written on Windows (since that's where Hammer works).
But it should work fine on Linux. To get dependencies:

1. You'll likely want a virtual environment to keep the packages isolated - see Python's `venv` module.
   Running `python -m venv some_folder/` will create one, then you can run the
   `activate` script inside there to enable the environment.
2. Run `python -m pip install -r requirements.txt` to install neceessary modules.
3. Optionally `python -m pip install -r test-requirements.txt` to run some test code.

### Building FGDs
FGDs are stored as individual files, in a [unified][unifiedfgd] format tagged with games. This
allows appropriate FGDs to be assembled for any Source game. To build an FGD for HL2 (for example):

```shell
cd src
python hammeraddons/unify_fgd.py export hl2 srctools -o "build/hl2.fgd"
```

Consult the lists at the start of the script for available tags. The first is the game/mod, any
additional ones are "features" like `srctools` for postcompiler features, or `propper` for those ents.

### Building from source

Many features require the postcompiler, which is a Python application. Releases have a compiled
build, but for development purposes it may be useful to build locally or run from source. This also
creates the `gen_choreo` utility.

Run `python -m PyInstaller postcompiler.spec` to freeze the application.
Optionally pass `--workpath XXX` and `--distpath XXX` to specify a temp folder and the destination
respectively.

The compiler can also be run from source by executing `hammeraddons/postcompiler.py`, 
with `src/` in `PYTHONPATH`.

[releases]: https://github.com/TeamSpen210/HammerAddons/releases
[installationwiki]: https://github.com/TeamSpen210/HammerAddons/wiki/Installation
[unifiedfgd]: https://github.com/TeamSpen210/HammerAddons/wiki/Unified-FGD
[skotty]: http://forums.thinking.withportals.com/downloads.php?view=detail&df_id=507
[tf2]: http://forums.tf2maps.net/showthread.php?t=4674
[ts2do]: http://halflife2.filefront.com/file/HalfLife_2_Upgraded_Base_FGDs;48139
[zps]: http://www.necrotalesgames.com/tools/index.php
[p1fgd]: https://cdn.discordapp.com/attachments/702167000574853242/917353063319212072/FGDs.zip
