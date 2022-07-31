
# Version (dev)


--------------------

# Version 2.5.0

* Fix two issues causing produced BSPs to be potentially corrupt. If your maps are mysteriously crashing on load, this may fix the issue.
* Particle systems will now be detected and packed along with their dependencies. This needs configuration in the config file.
* Optionally, the postcompiler can collapse and remove func_instance_io_proxy from maps entirely to save ents.
* Add comp_sequential_call: finds a sequence of entities (by distance or numeric suffix), then fires inputs delayed in order.
* Add comp_flicker: fires on/off and skin inputs repeatedly to simulate a flicker-on effect.
* comp_scriptvar_setter can now set global variables also.
* prop_paint_bomb will now show its collision mesh (futbols).
* Fix .ani files for models not being detected.
* Fix propcombine not working if blacklist is not set.
* Handle VPKs with non-ASCII bytes in filenames.
* BINK videos will now never be packed.
* When generating a default config, running from a sourcemod will be properly handled

--------------------

# Version 2.4.0
* Added comp_prop_rope_dynamic and comp_prop_cable_dynamic, for generating 3D ropes as dynamic props.
* Added comp_prop_rope_bunting, for positioning other props along a rope (could be used for lights, supports, decoration, etc).
* Prop ropes may optionally have a collision mesh.
* Added comp_propcombine_volume & tools/toolspropcombine, which allows specifing propcombine regions with brushwork. 
  This does leave remmnants of the brush in the map, so the point entity may still need to be used if near brush limits.
* Added comp_vactube_spline, which generates Portal 2 vactube models following a path.
* Add an editor model for decals, like overlays have.
* Added ability to specify rotation seed for vactubes.

--------------------

# Version 2.2.0 
* Add `comp_prop_rope`/`comp_prop_cable`: These allow generating 3D static prop versions of cables, like in Source 2. Place them down, choose a material, then connect them together like regular `move_rope`/`keyframe_rope`.
* The postcompiler is now able to properly handle pre-L4D entity outputs.
* Added "plugin" support to the compiler - directories can be specified which contain scripts to be run on the map in addition to the existing ones.
*  Vactubes now have a "next junction" option for manually specifying the next location, and have a 45 degree curve variant.
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
