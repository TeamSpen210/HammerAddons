"""Script to assist with merging Strata's fork back into upstream."""
from pathlib import Path
import io
import difflib

from srctools.fgd import EntityTypes, EntityDef, KVDef, KVOption
from srctools.filesys import RawFileSystem
from srctools import FGD
from sys import argv

MERGED = {  # Set of classnames we have checked already and know the diff is fine.
    "info_player_start", "env_soundscape", "momentary_rot_button",
    "ai_script_conditions", "logic_script", "trigger_hierarchy",
    "prop_under_button", "npc_bullseye", "paint_sphere",
    "damagetype", "env_sprite", "info_landmark_entry", "info_landmark_exit",
    "env_alyxemp", "env_laser", "logic_timer", 

    "scripted_scene", # Legacy entity

    "light", "light_spot", # I belive that in order to avoid confusion, we should
    # omit the dynamic settings on `light`

    "light_directional",
    "prop_portal", "prop_wall_projector", "prop_weighted_cube",


}

REPORT_DIR = Path('..', 'strata_merge').resolve()
MERGE_DIR = Path(REPORT_DIR, 'merged').resolve()

# Report things only missing in FGD2, not present in FGD2 but missing in FGD1
ONLY_MISSING_IN_FGD2 = True

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
    def mdirty(classname):
        nonlocal all_good
        if all_good:
            print(f"Classname: {classname}: \n")
            all_good = False

    def mdirty_internal1(msg):
        nonlocal all_good_internal1
        if all_good_internal1:
            print(msg)
            all_good_internal1 = False

    classes -= MERGED # remove merged

    for classname in classes:
        classname: str
        all_good = True

        if classname.startswith("comp_") or classname.startswith("base"): # Skip all postcomp entities and bases (?)
            continue

        try:
            pent: EntityDef = strata_fgd[classname]
        except KeyError:
            if ONLY_MISSING_IN_FGD2:
                continue

            mdirty()
            print("|-> Not present in FGD 1! Possibly removed?")
            continue

        try:
            hent: EntityDef = ha_fgd[classname]
        except KeyError:
            mdirty(classname)
            print("|-> New entity (not present in FGD 2)!")
            continue

        # KEYVALUES

        all_keys = set(pent.keyvalues.keys()) | set(hent.keyvalues.keys())
        
        for key in all_keys:
            all_good_internal1 = True


            # SPECIAL CASES

            # 1. linedivider_broken and linedivider_vscript are the same

            if key in ("linedivider_broken", "linedivider_vscript"):
                continue

            # END SPECIAL CASES

            try:
                pkv = pent.keyvalues[key]
            except KeyError:
                if ONLY_MISSING_IN_FGD2:
                    continue

                mdirty(classname)
                print(f"|-> Keyvalue '{key}' definition missing in FGD 1, not implemented?")
                continue

            try:
                hkv = hent.keyvalues[key]
            except KeyError:
                if key in ("comp_custom_model_type"):
                    continue

                mdirty(classname)
                print(f"|-> New keyvalue (not present in FGD 2): '{key}'")
                continue

            if len(pkv.keys()) != 1 or len(hkv.keys()) != 1:
                raise RuntimeError("You can only use already compiled (untagged) FGDs!")

            pkv: KVDef = list(pkv.values())[0]
            hkv: KVDef = list(hkv.values())[0]
            
            
            
            # Step 0 - Name checking - already done
            # Step 1 - Kv type
            if pkv._type != hkv._type:
                mdirty(classname)
                mdirty_internal1(f"|-> Keyvalue '{key}': ")
                print(f"    |-> Different kv type | FGD 1: {pkv._type} <=> FGD 2: {hkv._type}")

            # Step 2 - Default value
            if pkv.default != hkv.default:
                if key in ("fademaxdist", "fademindist", "fadescale", "modelscale", "texframeindex", "shadowcastdist"):
                    continue

                mdirty(classname)
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
                    if ch == "": # I'm not sure what is happening here, but sometimes we get issues where a value like this shows up
                        continue

                    try:
                        pkvch = pkv_choices_d[ch]
                    except KeyError:
                        if ONLY_MISSING_IN_FGD2:
                            continue
                        
                        mdirty(classname)
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] Value {ch} does not exist in FGD 1!")
                        continue

                    try:
                        hkvch = hkv_choices_d[ch]
                    except KeyError:
                        mdirty(classname)
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] New value: '{ch}' -> '{pkvch.name}'")
                        continue

                    if pkvch.name != hkvch.name:

                        # Special cases: Some kvs have their names changed (like grammatical changes)
                        if key in ("renderfx", "mincpulevel", "mingpulevel", "maxgpulevel", "maxcpulevel", "teamnum"):
                            continue

                        mdirty(classname)
                        mdirty_internal1(f"|-> Keyvalue '{key}': ")
                        print(f"    |-> [Choices] Different names for value '{ch}' | FGD 1: '{pkvch.name}' | FGD 2: '{hkvch.name}' ")

        # End: keyvalues

        # INPUTS

        all_inputs_names = set(pent.inputs.keys()) | set(hent.inputs.keys())
        for inp in all_inputs_names:

            if not inp in pent.inputs.keys():
                if ONLY_MISSING_IN_FGD2:
                    continue

                mdirty(classname)
                print(f" |-> Missing input in FGD 1: {inp}")
                continue

            if not inp in hent.inputs.keys():
                mdirty(classname)
                print(f"|-> New input: {inp}")

        all_outputs_names = set(pent.outputs.keys()) | set(hent.outputs.keys())
        for inp in all_outputs_names:

            if not inp in pent.outputs.keys():
                if ONLY_MISSING_IN_FGD2:
                    continue

                mdirty(classname)
                print(f" |-> Missing output in FGD 1: {inp}")
                continue

            if not inp in hent.outputs.keys():
                mdirty(classname)
                print(f"|-> New output: {inp}")
                

        print("\n")
        



if __name__ == '__main__':
    main()
