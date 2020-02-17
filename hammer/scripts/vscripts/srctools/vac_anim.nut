// Implements vactube animation with randomised data.
// Compiler appends "anim_xx <- anim(...)" and "obj_xx <- obj(...)" after this to populate tables.

// A thing in the tube, which might be put in the dropper.
class Cargo {
	cube_model = ""; // The real physics model used for the prop_weighted_cube.

    model = "";  // model to use for the fake "cube".
    skin = "0"; // Skin to use.
    localpos = "0 0 0"; // Local offset.
    tv_skin = 0; // Skin on the Diversity Scanner TV.
    constructor (vac_mdl, cube_mdl, off, tv) {
		model = vac_mdl;
		cube_model = cube_mdl;
		localpos = off;
		tv_skin = tv;
    }
	function _tostring () {
	    return "<Cube \"" + model + "\", weight=" + weight + ", zoff=" + z_offset + ">";
	}
}

// Nodes with outputs need to be delayed.
class Output {
	time = 0.0; // Delay after anim start before it should fire.
	target = null; // The node to fire inputs at.
	scanner = null; // If set, the scanner to fire skin inputs to.
	constructor (_time, node_name, scanner_name) {
		time = _time;
		target = Entities.FindByName(null, node_name);
		if (scanner_name != null) {
		scanner = Entities.FindByName(null, scanner_name);
		} else {
			scanner = null;
		}
	}
}

// A specific animation/route.
class Anim {
	name = ""; // Name of the animation.
	cargo_type = null; // Cube type it spawns.
	req_spawn = false; // If the dropper needs a new cube.
	pass_io = []; // Outputs to delay-fire.
	duration = 0.0; // Length of animation.

	constructor (anim_name, time, cube_type, pass_io_lst) {
		name = anim_name;
		duration = time;
		cargo_type = cube_type;
		req_spawn = false;
		pass_io = pass_io_lst;
	}
	function tostring() {
	    return "<Anim \"" + name + "\", " + duration + "s, type = " + cargo_type + ", reqesting spawn=" + req_spawn + ">";
	}
}

// Holds the ents which are already in the map or are being replaced.
class EntSet {
	reuse_time = 0.0;
	mover = null;
	visual = null;
	constructor (time, mov, vis) {
		reuse_time = time;
		mover = mov;
		visual = vis;
	}
}

// Animations which go to a dropper, and ones that just go to deco.
ANIM_DROP <- [];
ANIM_DECO <- [];

CARGOS <- [];

// The list of every vactube ent in the map. We use this to allow recycling old ones.
if (!("vactube_objs" in getroottable())) {
	::vactube_objs <- [];
}

function show() {
    foreach (anim in ANIM_DROP) {
        printl("Drop: " + anim.tostring());
    }
    foreach (anim in ANIM_DECO) {
        printl("Deco: " + anim.tostring());
    }
}

// Helper functions to create and register the types.
function obj(vac_mdl, cube_mdl, weight, off, tv) {
	local cargo = Cargo(vac_mdl, cube_mdl, off, tv);
    for (local i = 0; i < weight; i++) {
    	CARGOS.append(cargo);
    }
	return cargo;
}
function anim(anim_name, time, type, pass_io_lst) {
	local ani = Anim(anim_name, time, type, pass_io_lst);
	if (type == null) {
		ANIM_DECO.append(ani);
	} else {
		ANIM_DROP.append(ani);
	}
	return ani;
}

// Spawn a new cube, or recycle a new one.
function make_cube() {
    local anim = null;
    local cargo_type;
    foreach (drop_anim in ANIM_DROP) {
        if (drop_anim.req_spawn) {
    		cargo_type = drop_anim.cargo_type;
    		drop_anim.req_spawn = false;
    		anim = drop_anim;
    		break;
        }
    }

    if (anim == null) {
		// No active droppers, spawn a random object and go to a random destination.
		if (ANIM_DECO.len() == 0) {
			return; // No positions...
		}
		anim = ANIM_DECO[RandomInt(0, ANIM_DECO.len()-1)];
		if (RandomInt(1, 1000) == 2) {
			cargo_type = CHICKEN;
		} else {
			cargo_type = CARGOS[RandomInt(0, CARGOS.len()-1)];
		}
	}


	// Now either find an existing cargo we can use or create a new one.
	local cargo = null;
	local cur_time = Time();
	foreach (poss_cargo in ::vactube_objs) {
	    if (cur_time > poss_cargo.reuse_time) {
    		cargo = poss_cargo;
    		break;
	    }
	}
	if (cargo == null) {
		// Need to spawn a new one.
		self.SpawnEntity();
		local visual = Entities.FindByNameWithin(null, "_vactube_temp_visual", self.GetOrigin(), 16);
   		local mover = Entities.FindByNameWithin(null, "_vactube_temp_mover", self.GetOrigin(), 16);

	    // Rename so we don't detect this again.
	    EntFireByHandle(mover, "AddOutput", "targetname _vactube_mover", 0, self, self);
	    EntFireByHandle(visual, "AddOutput", "targetname _vactube_visual", 0, self, self);

	    // Then add to our total queue. As a safeguard, init with reuse time only a bit after
	    // now so if this one crashes another can reuse it.
		cargo = EntSet(cur_time + 3.0, mover, visual);
		::vactube_objs.append(cargo);
		// For tracking spawning, set this.
		printl("Vactube ent count: " + (::vactube_objs.len() * 2).tostring());
	}

	cargo.visual.SetModel(cargo_type.model);
    EntFireByHandle(cargo.visual, "Skin", cargo_type.skin, 0, self, self);
    EntFireByHandle(cargo.visual, "EnableDraw", "", 0, self, self);
    EntFireByHandle(cargo.visual, "SetLocalOrigin", cargo_type.localpos, 0, self, self);
    EntFireByHandle(cargo.mover, "SetAnimation", anim.name, 0, self, self);
    EntFireByHandle(cargo.visual, "DisableDraw", "", anim.duration, self, self);
    cargo.reuse_time = cur_time + anim.duration + 0.1; // Make sure enable/disable inputs don't get mixed up.

    foreach (pass_out in anim.pass_io) {
		// printl("Output: " + pass_out.target +  " @ " + pass_out.time);
		EntFireByHandle(pass_out.target, "FireUser4", "", pass_out.time, self, self);
		if (pass_out.scanner != null && cargo_type.tv_skin != 0) {
			EntFireByHandle(pass_out.scanner, "Skin", cargo_type.tv_skin.tostring(), pass_out.time, self, self);
		}
    }
}
