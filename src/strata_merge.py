"""Script to assist with merging Strata's fork back into upstream."""
from pathlib import Path
import io
import difflib

from srctools.fgd import EntityTypes, EntityDef, KVDef, KVOption
from srctools.filesys import RawFileSystem
from srctools import FGD
from sys import argv

MERGED = {  # Set of classnames we have checked already and know the diff is fine.
    'baseentityvisbrush', 'baselight', 'baseentityinputs', 'baseentityphysics', 'basepropphysics',
    'fadedistance', 'gibshooterbase', 'grenadeuser', 'basebeam', 'baselogicalnpc', 'basenpc',
    'basespotlight', 'button', 'door', 'followgoal', 'hintnode',
    'baseheadcrab', 'basehelicopter', 'baseportbutton', 'basetrain', 'combinescanner', 'damagetype',

    'ai_changetarget', 'ai_goal_actbusy', 'ai_goal_injured_follow', 'scripted_scene',
    'ai_script_conditions', 'aiscripted_schedule', 'ambient_generic', 'keyframe_track',
    'monster_generic', 'move_keyframed', 'bounce_bomb', 'func_instance_origin', 'test_sidelist',
    'test_traceline', 'color_correction_volume', 'combine_mine', 'cycler', 'game_weapon_manager',
    'color_correction', 'game_score', 'generic_actor', 'grenade_helicopter',
    'comp_adv_output', 'comp_case', 'comp_entity_finder', 'comp_kv_setter', 'comp_prop_cable',
    'comp_prop_cable_dynamic', 'comp_prop_rope', 'comp_prop_rope_dynamic', 'comp_propcombine_set',
    'comp_propcombine_volume', 'comp_relay', 'comp_sequential_call', 'comp_trigger_coop',
    'comp_vactube_start', 'comp_vactube_junction', 'comp_vactube_end', 'comp_vactube_sensor',
    'comp_vactube_spline', 'comp_vactube_object', 'comp_piston_platform', 'comp_movie_fitter',
    'comp_multi_command', 'comp_trigger_p2_goo', 'hammer_model', 'hammer_notes',
    'env_effectscript', 'env_fade', 'env_fire', 'env_fog_controller',
    'env_projectedtexture', 'env_portal_laser', 'env_soundscape', 'env_rockettrail', 'env_sun',
    'env_rotorwash_emitter', 'env_blood', 'env_bubbles', 'env_embers', 'env_starfield', 'env_steam',
    'env_zoom', 'env_ar2explosion', 'env_explosion', 'env_shooter', 'env_smoketrail', 'env_speaker',
    'env_sprite', 'env_tonemap_controller', 'env_wind', 'env_microphone', 'env_movieexplosion',
    'env_global', 'env_headcrabcanister', 'env_screenoverlay', 'env_player_viewfinder',
    'env_particlescript', 'env_credits',
    'hot_potato', 'hot_potato_catcher', 'hot_potato_socket', 'hot_potato_spawner',
    'filter_enemy', 'filter_multi', 'func_areaportal', 'func_breakable', 'func_breakable_surf',
    'filter_activator_class', 'filter_activator_involume', 'filter_damage_type',
    'filter_activator_team', 'filter_activator_keyfield', 'filter_activator_model',
    'func_brush', 'func_combine_ball_spawner', 'func_door', 'func_illusionary', 'func_physbox',
    'func_instance', 'func_instance_io_proxy', 'func_clip_vphysics', 'func_tankairboatgun',
    'func_movelinear', 'linked_portal_door', 'func_smokevolume', 'func_tanklaser', 'func_tankmortar',
    'func_tankphyscannister', 'func_door_rotating', 'func_placement_clip', 'func_proprrespawnzone',
    'func_precipitation', 'func_water_analog', 'func_wall_toggle', 'func_wall', 'func_water',
    'func_dustcloud', 'func_dustmotes',
    'game_globalvars', 'game_player_team', 'game_text',
    'item_ammo_357', 'item_ammo_357_large',
    'item_ammo_ar2', 'item_ammo_ar2_altfire', 'item_ammo_ar2_large', 'item_healthcharger',
    'item_healthkit', 'item_healthvial', 'item_item_crate', 'item_large_box_lrounds',
    'item_large_box_mrounds', 'item_large_box_srounds', 'item_rpg_round', 'item_suit',
    'item_suitcharger', 'item_box_buckshot', 'item_battery', 'item_grubnugget',
    'item_ammo_crossbow', 'item_ammo_crate', 'item_ammo_pistol', 'item_ammo_pistol_large',
    'item_ammo_smg1', 'item_ammo_smg1_grenade', 'item_ammo_smg1_large', 'item_ar2_grenade',
    'info_coop_spawn', 'info_apc_missile_hint', 'info_ladder_dismount',
    'info_lighting_relative', 'info_npc_spawn_destination', 'info_overlay_transition',
    'item_box_lrounds', 'item_box_mrounds', 'item_box_srounds', 'info_paint_sprayer',
    'info_radar_target', 'info_snipertarget', 'info_target_gunshipcrash', 'info_overlay',
    'info_player_deathmatch', 'physics_cannister', 'point_bugbait', 'point_worldtext',
    'info_target_vehicle_transition', 'info_target_helicopter_crash', 'info_teleporter_countdown',
    'info_darknessmode_lightsource', 'info_intermission', 'infodecal',
    'logic_achievement', 'logic_choreographed_scene', 'logic_compare', 'logic_playerproxy',
    'logic_timer', 'logic_auto', 'logic_convar', 'logic_gate', 'logic_measure_direction',
    'logic_measure_movement', 'logic_modelinfo', 'logic_playmovie', 'logic_scene_list_manager',
    'logic_script', 'logic_sequence', 'material_modify_control', 'logic_case',
    'logic_eventlistener_itemequip', 'logic_relay',
    'light_dynamic', 'light_directional',
    'light', 'light_spot', 'light_rt', 'light_rt_spot', 'light_environment', 'logic_random_outputs',
    'prop_thumper', 'path_track', 'prop_laser_catcher', 'prop_testchamber_sign', 'prop_tractor_beam',
    'prop_testchamber_door', 'paint_sphere', 'prop_door_rotating', 'prop_static', 'prop_portal',
    'prop_dynamic_ornament', 'prop_glados_core', 'prop_physics_ragdoll', 'prop_stickybomb',
    'point_bonusmaps_accessor', 'point_broadcastclientcommand', 'point_camera', 'point_changelevel',
    'point_teleport', 'point_viewcontrol', 'point_viewcontrol_multiplayer', 'point_spotlight',
    'npc_pigeon', 'npc_crow', 'npc_rollermine', 'npc_seagull', 'npc_security_camera',
    'npc_strider', 'npc_zombine', 'npc_dog', 'npc_eli', 'npc_advisor', 'npc_citizen', 'npc_alyx',
    'npc_clawscanner', 'npc_combine_s', 'npc_combinedropship', 'npc_combinegunship', 'npc_cscanner',
    'npc_enemyfinder', 'npc_enemyfinder_combinecannon', 'npc_fastzombie', 'npc_fastzombie_torso',
    'npc_heli_avoidsphere', 'npc_headcrab_poison', 'npc_headcrab_black', 'npc_grenade_frag',
    'npc_helicopter', 'npc_metropolice', 'npc_antlion_template_maker', 'npc_bullseye', 'npc_monk',
    'npc_combine_camera', 'npc_combine_cannon', 'npc_hunter_maker', 'npc_launcher', 'npc_tripmine',
    'npc_personality_core', 'npc_portal_turret_floor', 'npc_rocket_turret', 'npc_template_maker',
    'trigger_soundoperator', 'trigger_soundscape', 'trigger_playermovement', 'trigger_playerteam',
    'trigger_togglesave', 'trigger_remove', 'trigger_rpgfire', 'trigger_tonemap', 'trigger_transition',
    'trigger_wind', 'trigger_weapon_dissolve', 'trigger_weapon_strip', 'trigger_portal_cleanser',
    'trigger_proximity', 'trigger_serverragdoll', 'trigger_setspeed', 'trigger_teleport',
    'trigger_waterydeath',
    'vgui_movie_display', 'weapon_portalgun', 'momentary_rot_button',
}

REPORT_DIR = Path('..', 'strata_merge').resolve()
MERGE_DIR = Path(REPORT_DIR, 'merged').resolve()


def main() -> None:
    """Check all the FGDs."""
    if len(argv) < 3:
        raise RuntimeError("Please specify paths to the base FGD (P2CE) and compare with FGD (HA).")

    path1 = Path(argv[1])
    path2 = Path(argv[2])
    fsys = RawFileSystem(path1.parent)
    strata_fgd = FGD()
    strata_fgd.parse_file(fsys, fsys[path1.name], encoding='iso-8859-1')
    fsys = RawFileSystem(path1.parent)
    ha_fgd = FGD()
    ha_fgd.parse_file(fsys, fsys[path2.name], encoding='iso-8859-1')

    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in REPORT_DIR.iterdir():
        if fname != MERGE_DIR:
            fname.unlink()
    for fname in MERGE_DIR.iterdir():
        fname.unlink()

    (REPORT_DIR / '.gitignore').write_text('*')

    bases = {
        ent.classname.casefold()
        for ent_list in [strata_fgd, ha_fgd]
        for ent in ent_list
        if ent.type is EntityTypes.BASE
    }

    # We're comparing strictly the attributes of entities
    # Defining which property belongs to which base should be determined by the merger
    ha_fgd.collapse_bases()
    strata_fgd.collapse_bases()

    classes = set(strata_fgd.entities.keys()) | set(ha_fgd.entities.keys())
    print(f'{len(classes)} entities defined, {len(MERGED)} suppressed.')
    #strata_master = strata_fgd['masterent']
    added = []
    removed = []
    count = 0

    all_good = True
    all_good_internal1 = True # Used for comparing internal data

    #Short aliases for less typing
    def mdirty():
        nonlocal all_good
        if all_good:
            print("\n")
            all_good = False

    def mdirty_internal1(msg):
        nonlocal all_good_internal1
        if all_good_internal1:
            print(msg)
            all_good_internal1 = False

    for classname in classes:
        classname: str
        all_good = True
        print(f"Checking classname: {classname} ... ", end="")

        try:
            pent: EntityDef = strata_fgd[classname]
        except KeyError:
            if classname.startswith("comp_"):
                print("New postcompiler entity skipped! All Good!")
                continue

            mdirty()
            print("|-> Not present in FGD 1! Possibly removed?")
            continue

        try:
            hent: EntityDef = ha_fgd[classname]
        except KeyError:
            mdirty()
            print("|-> New entity (not present in FGD 2)!")
            continue

        # KEYVALUES

        all_keys = set(pent.keyvalues.keys()) | set(hent.keyvalues.keys())
        
        for key in all_keys:
            all_good_internal1 = True

            try:
                pkv = pent.keyvalues[key]
            except KeyError:
                mdirty()
                print(f"|-> Keyvalue '{key}' definition missing in FGD 1, not implemented?")
                continue

            try:
                hkv = hent.keyvalues[key]
            except KeyError:
                mdirty()
                print(f"|-> New keyvalue (not present in FGD 2): '{key}'")
                continue

            if len(pkv.keys()) != 1 or len(hkv.keys()) != 1:
                raise RuntimeError("You can only use already compiled (untagged) FGDs!")

            pkv: KVDef = list(pkv.values())[0]
            hkv: KVDef = list(hkv.values())[0]
            
            
            
            # Step 0 - Name checking - already done
            # Step 1 - Kv type
            if pkv._type != hkv._type:
                mdirty()
                mdirty_internal1(f"|-> Keyvalue '{key}': ")
                print(f"    |-> Different kv type | FGD 1: {pkv._type} <=> FGD 2: {hkv._type}")

            # Step 2 - Default value
            if pkv.default != hkv.default:
                mdirty()
                mdirty_internal1(f"|-> Keyvalue '{key}': ")
                print(f"    |-> Different default value | FGD 1: {pkv.default} <=> FGD 2: {hkv.default}")

            # Step 3 - Choices
            pkv_choices: list[KVOption] = pkv.options
            hkv_choices: list[KVOption] = hkv.options

            if pkv_choices is not None and hkv_choices is not None:
                pkv_choices_d = {x.value: x for x in pkv_choices}
                hkv_choices_d = {x.value: x for x in hkv_choices}

                allchoices = set([x.value for x in pkv_choices]) | set([x.value for x in hkv_choices])

                for ch in allchoices:
                    try:
                        pkvch = pkv_choices_d[ch]
                    except KeyError:
                        mdirty()
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] Value {ch} does not exist in FGD 1!")
                        continue

                    try:
                        hkvch = hkv_choices_d[ch]
                    except KeyError:
                        mdirty()
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] New value: '{ch}' -> '{pkvch.name}'")
                        continue

                    if pkvch.name != hkvch.name:
                        mdirty()
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] Different names for value '{ch}' | FGD 1: '{pkvch.name}' | FGD 2: '{hkvch.name}' ")

        # End: keyvalues

        # INPUTS

        all_inputs_names = set(pent.inputs.keys()) | set(hent.inputs.keys())
        for inp in all_inputs_names:

            if not inp in pent.inputs.keys():
                mdirty()
                print(f" |-> Missing input in FGD 1: {inp}")
                continue

            if not inp in hent.inputs.keys():
                mdirty()
                print(f"|-> New input: {inp}")

        all_outputs_names = set(pent.outputs.keys()) | set(hent.outputs.keys())
        for inp in all_outputs_names:

            if not inp in pent.outputs.keys():
                mdirty()
                print(f" |-> Missing output in FGD 1: {inp}")
                continue

            if not inp in hent.outputs.keys():
                mdirty()
                print(f"|-> New output: {inp}")
                


        if all_good:
            print("All Good!")

        print("\n")
        



if __name__ == '__main__':
    main()
