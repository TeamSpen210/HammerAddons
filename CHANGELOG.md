# Version Dev

# Enhancements
* Added textures for 4 rendertargets (camera, water reflection/refraction, full framebuffer). These give a nice preview in Hammer.
* Add ability to randomise output delay to `comp_relay` and `comp_adv_output`.
* Added `comp_vactube_sensor`, which allows detecting the presence of vactube objects.
* Allow configuring various shadow/fast reflection options for vactube object ents.


--------------------


# Version 2.5.3

## Enhancements
* #163: Added `comp_adv_output`, which allows adding a single output with complex behaviour.
* Added `comp_case`, a version of `logic_case` that is collapsed into callers like `comp_relay`.
* Vactube props can now be generated with glass/frames seperately, and with bezier curves instead of catmul-romm. (Authored by [asd417](https://github.com/asd417)) 
* Added `--verbose` parameter, for showing DEBUG messages.
* Added `--regenerate` parameter, to force all models to be regenerated from scratch.
* Added ability to specify alt skins when using `comp_prop_cable_dynamic`.
* Weapon scripts are now packed along with the models/sounds they use.
* #210: Add `OnFinished` output to `comp_numeric_transition`.
 Add an option to specify the maximum distance for automatic combined props.
* Allow combining models containing `$collisionjoints`.
* Add missing `bunting` keyvalue to `comp_prop_cable`.
* Areaportal windows will automatically force the brushes used to nonsolid, and clear some physics data. 
* Propcombine will no longer merge props found in different areaportal areas. This allows props on the outside of a building to be culled when inside, or vice versa.
* Automatically set the "transmit to client" flag for `info_target`s used as particle system destinations.
* Change `sky_camera` model to be more visible with `tools/toolsskybox` behind it.
* Allow physboxes to completely override their mass.
* The postcompiler can now automatically handle custom models for various Portal 2 entities.
## Bugfixes
* `comp_pack_rename` will now automatically also include the auxiliary `.mdl` files.
* #10: The center of the axis helper used for sprites can now be clicked on.
* #232: Readd missing `OnFizzled` output on `prop_weighted_cube`.
* Limit the size of propcombined groups to avoid hitting vertex limits.
* Prevent automatically packing models specified only as Hammer previews in various entities.
* Fix propcombine sometimes removing collisions entirely from component props.
* Add an option to allow the sources for compiled models to be preserved.
* Improve description of `wait` and `OnTrigger` options in `trigger_multiple`.
* Fix #192: Use both specified model and cube type field to find matching cubes for vactubes.
* Restore missing projected texture shadow keyvalues.
* Fix `comp_precache_sound` not handling sound characters at the start of raw filenames.
* Allow `comp_vactube_start` to be set to have a timer which starts disabled.
* If required, add various QC flags like `$mostlyopaque` to propcombined props.
* Set Precache/OnPostSpawn in generated VScript, to prevent double-firing functions already in Entity Scripts.

--------------------

# Version 2.5.2
* Added new `srctools_paths.vdf` config, allowing specifying the base location for all paths in the main config. 
  This separation allows mod projects to synchronise the main config, while customising the new path config for installation locations on each computer.
* Fix `scriptlist` being used in pre-L4D2 branches.
* Fix an issue causing a single brush face in the map to be "missing" a triangle.
* Add some support for Portal 2 Educational Version and The Stanley Parable (via @Adrthegamedev).
* Clarify that Portal 2 uses `maps/map_name_particles.txt`.
* Ensure culled bases are replaced by their own bases (fixes `Item`, for example).
* Update to Mapbase 7.1, and add Entropy Zero 2 entities.
* Added icon for `npc_heli_avoidsphere`.
* Change propcombine/geocable to log just the number of culled models, not every single file.
* TF2 now has VScript support, updated FGDs to include the corresponding options.
* Added new blocklist/allowlist options to permit fine-grained control of which files get packed.
  * A specific option was added to strip existing cubemap files from the map, if this is necessary to fix crashes while building cubemaps.
* Add transform to allow using `/n` in `game_text` messages to produce multiline ones.
* #167: Add `comp_player_input_helper`, an entity to trick Hammer into autocompleting player inputs.
* Transforms should now use `EntityDef.engine_def()`, not `ctx.fgd` to allow partially parsing the database.

--------------------

# Version 2.5.1
* Tweak `comp_prop_rope` and `comp_vactube_junction` "Group" descriptions to make it clear they're optional.
* Tweak `comp_vactube_junction` "Override" description to make it clear they're optional.
* Tweak `comp_propcombine_set`/`_volume` "Model Filter" description to make it more clear.
* Add additional log messages for propcombine.
* Add `comp_relay`'s "Control Type"/"Control Value" option to most comp_ entities, to allow disabling them via fixup values.
* Make `point_viewcontrol` defaults to be more useful.
* Change the editor model for `prop_testchamber_door` to use a dashed line.
* The config format for additional plugins has changed - an ID is now required, allowing for a consistent import path to be used.
* Fix issues where `comp_flicker` may run indefinitely.
* Fix the `use_comma_sep` config option.
* Compilation of prop ropes, propcombine models, and decompilation will now all be done concurrently, using all CPU cores.
* Added a config option (`propcombine_pack`) to control whether propcombined models will be packed.
* Tweak the distance `comp_trigger_p2_goo` places the physics trigger below the surface.
* Correctly handle "only once" when collapsing outputs in entities such as `comp_relay`.
* Improve matching behaviour for the Entity Handle mode in `comp_scriptvar_setter`, and add a `Qangle()` mode for L4D2/Mapbase.
* Ensure propcombine entities are deleted from the BSP in all cases.
* The generated vactube animation prop is now textured with a valid material.
* New entity sprites: `npc_vehicledriver`, `comp_numeric_transition`, `point_broadcastclientcommand`.
* (#154): Add scale keyvalue to hammer_notes.
* (#42): Add editor models for Black Mesa health and suit chargers.
* (#76): Make all weapon entities include `CBaseAnimating` I/O and keyvalues.
* (#120): Fix `env_bubbles`, `env_embers`, `func_precipitation` and `func_smokevolume` having an `origin` keyvalue. These entities break if their origin is not `0 0 0`.
* Indicate the allowed combine ball sizes - `1-12`.
* Snap propcombine props to within 45 degrees, not 15.
* Remove `--showgroups` command line option. Source provides the `r_staticpropinfo` convar which performs the same function.
* Use a cache file to avoid needing to reparse particle system files every run.

--------------------

# Version 2.5.0
* Fix two issues causing produced BSPs to be potentially corrupt. If your maps are mysteriously crashing on load, this may fix the issue.
* Particle systems will now be detected and packed along with their dependencies. 
  This needs configuration in the config file, since different games use different filenames.
* Optionally, the postcompiler can collapse and remove `func_instance_io_proxy` from maps entirely to save ents.
* Add `comp_sequential_call`: finds a sequence of entities (by distance or numeric suffix), then fires inputs delayed in order.
* Add `comp_flicker`: fires on/off and skin inputs repeatedly to simulate a flicker-on effect.
* `comp_scriptvar_setter` can now set global variables also.
* `prop_paint_bomb` will now show its collision mesh (futbols).
* Fix .ani files for models not being detected.
* Fix propcombine not working if blacklist is not set.
* Handle VPKs with non-ASCII bytes in filenames.
* BINK videos will now never be packed.
* When generating a default config, running from a sourcemod will be properly handled.

--------------------

# Version 2.4.0
* Added `comp_prop_rope_dynamic` and `comp_prop_cable_dynamic`, for generating 3D ropes as dynamic props.
* Added `comp_prop_rope_bunting`, for positioning other props along a rope (could be used for lights, supports, decoration, etc).
* Prop ropes may be configured to generate a collision mesh.
* Added `comp_propcombine_volume` & `tools/toolspropcombine`, which allows specifing propcombine regions with brushwork. 
  This does leave remmnants of the brush in the map, so the point entity may still need to be used if near brush limits.
* Added `comp_vactube_spline`, which generates Portal 2 vactube models following a path.
* Add an editor model for decals, like overlays have.
* Added ability to specify rotation seed for vactubes.

--------------------

# Version 2.2.0 
* Add `comp_prop_rope`/`comp_prop_cable`: These allow generating 3D static prop versions of cables, like in Source 2. 
  Place them down, choose a material, then connect them together like regular `move_rope`/`keyframe_rope`.
* The postcompiler is now able to properly handle pre-L4D entity outputs.
* Added "plugin" support to the compiler - directories can be specified which contain scripts to be run on the map in addition to the existing ones.
* Vactubes now have a "next junction" option for manually specifying the next location, and have a 45 degree curve variant.
* Add a pile of new entity sprites by @lazyRares.
* Propcombine can now use a bundled copy of Crowbar to decompile models if the sources are not available.

--------------------

# Version 2.1.6
* Add a set of cubedropper instances for Portal 2.
* Vactube enhancements:
	* cross-junction splitter for vactubes
	* Support for frankenturrets
	* Allow having different object sets for each vactube
* Add `comp_relay`, a simplified version of relays which is collapsed into the callers.
* Update to Mapbase 4.1
* Add `UniqueState`  inputs to `logic_branch_listener`, which creates a `logic_branch` for each unique input entity.
* Sprites for `skybox_swapper`, `comp_pack_replace_soundscript`, `env_portal_credits` and `info_ping_detector` by Luke.

--------------------

# Version 2.1.5
* Fix triggers not having a "Clients" spawnflag
* Add the vactube dropper instance
* Make the postcompiler more forgiving - it will now skip soundscripts it can't find/parse.

--------------------

# Version 2.1.4
* Merge in Black Mesa and latest Mapbase changes.
* Add in vastly expanded visgroup sets for most entities.
* Add a lot of internal or otherwise hidden entities, improve HL2 related support.
* Add a lot of entities to internal packing database, allowing packing resources these use in code.
* Fix vactube system not working properly with multiple source points.
* Fix some incorrect math in `comp_propcombine_set` shape checks.
* Add `--showgroups` option, which randomly tints each propcombine group to let you see which props have been combined.

--------------------

# Version 2.1.3
- Fix vactube system not functioning when only decorative objects are provided.
- Fix packing system always not packing from VPKs
- Fix `logic_player_slowtime` appearing in all games.
- Set `trigger_portal_cleanser` to default to clients and physics.
- Add some new options to `comp_trigger_p2_goo`, designed for bottomless pits.
- Add icon for `comp_entity_mover`.
- Automatically detect `pak02_dir.vpk` etc folders.

--------------------

# Version 2.1.1
- Upload the correct compiler version.

--------------------

# Version 2.1.0
* Add `comp_numeric_transition`: this allows transitioning an entity option gradually from one value to another over time.
* Add `comp_scriptvar_setter`: a tool for copying positions or similar configuration to a VScript instance.
* For Portal 2, add a vactube animation system for building decorative tube networks.
* Fix a few icons being missing.

--------------------

# Version 2.0.2
* Fix `func_dustmotes` having extra keyvalues which break it.
* Fix several classnames being present or absent which should not.
* Add back `rendermode`/`rendercolor` to `env_lightglow`, `env_sprite*`, `env_spritetrail`, `env_steam` and `env_sun`.
* Add "Invert" option to `comp_kv_setter`.
* Make `move_rope` and `keyframe_rope` interchangeable/identical, as they are ingame.

* Fix Portal 1's Hammer having a few syntax errors.

--------------------

# Version 2.0.1
* Fix an issue where brushes have stray bounding boxes and may break maps.

-------------------

# Version 2.0.0
* Combine FGD files from all engine branches into a tagged, unified database.
* Implement postcompiler allowing many features including auto-packing and static prop combine
* Add lots of new editor sprites.
