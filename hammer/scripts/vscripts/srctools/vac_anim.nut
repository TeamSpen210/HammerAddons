// Implements vactube animation with randomised data.
// Compiler appends "anim_xx <- anim(...)" and "obj_xx <- obj(...)" after this to populate tables.

// A thing in the tube, which might be put in the dropper.
class Cargo {
	cube_model = ""; // The real physics model used for the prop_weighted_cube.

    model = "";  // model to use for the fake "cube".
    skin = 0; // Skin to use.
    localpos = "0 0 0"; // Local offset.
    tv_skin = 0; // Skin on the Diversity Scanner TV.
    constructor (vac_mdl, vac_skin, cube_mdl, off, tv) {
		model = vac_mdl;
		skin = vac_skin;
		cube_model = cube_mdl;
		localpos = off;
		tv_skin = tv;
    }
	function _tostring() { return tostring() }
	function tostring() {
	    return "<Cargo \"" + model + "#" + skin + "\", cube=" + cube_model + ", off=" + localpos + ", tv_skin=" + tv_skin + ">";
	}
}

// Nodes with outputs need to be delayed.
class Output {
	time = 0.0; // Delay after anim start before it should fire.
	target = null; // The node to fire inputs at.
	scanner = null; // If set, the scanner to fire skin inputs to
	// The magnitude is the time taken to travel HALF of a tv scanner prop. If negative, the user wants
	// it to stay on until another cube arrives.
	tv_offset = 0.0; 
	constructor (_time, node_name, scanner_name, _tv_offset=0.1) {
		time = _time;
		tv_offset = _tv_offset;
		target = Entities.FindByName(null, node_name);
		if (scanner_name != null) {
		scanner = Entities.FindByName(null, scanner_name);
		} else {
			scanner = null;
		}
	}
	function _tostring() { return tostring() }
	function tostring() {
	    return "<Output @" + time + ", targ: \""+target.GetName() + "\", scanner=" + scanner + ">";
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
	function _tostring() { return tostring() }
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
	function _tostring() { return tostring() }
	function tostring() {
	    return "<EntSet time=" + reuse_time + ">";
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
function obj(vac_mdl, vac_skin, cube_mdl, weight, off, tv) {
	local cargo = Cargo(vac_mdl, vac_skin, cube_mdl, off, tv);
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
		cargo_type = CARGOS[RandomInt(0, CARGOS.len()-1)];
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
	} else {
		// Teleport to the right position.
		cargo.mover.SetAbsOrigin(self.GetOrigin());
	}

	cargo.visual.SetModel(cargo_type.model);
    EntFireByHandle(cargo.visual, "Skin", cargo_type.skin.tostring(), 0, self, self);
    EntFireByHandle(cargo.visual, "EnableDraw", "", 0, self, self);
    EntFireByHandle(cargo.visual, "SetLocalOrigin", cargo_type.localpos, 0, self, self);
    EntFireByHandle(cargo.mover, "SetAnimation", anim.name, 0, self, self);
    EntFireByHandle(cargo.visual, "DisableDraw", "", anim.duration, self, self);
    cargo.reuse_time = cur_time + anim.duration + 0.1; // Make sure enable/disable inputs don't get mixed up.

    foreach (pass_out in anim.pass_io) {
		EntFireByHandle(pass_out.target, "FireUser4", "", pass_out.time, self, self);
		if (pass_out.scanner != null && cargo_type.tv_skin != 0) {
			if (pass_out.tv_offset < 0) { // Just fire on entry.
				EntFireByHandle(pass_out.scanner, "Skin", cargo_type.tv_skin.tostring(), pass_out.time + pass_out.tv_offset, self, self);
			} else { 
				// Fire on entry, then reset after.
				EntFireByHandle(pass_out.scanner, "Skin", cargo_type.tv_skin.tostring(), pass_out.time - pass_out.tv_offset, self, self);
				EntFireByHandle(pass_out.scanner, "Skin", "0", pass_out.time + pass_out.tv_offset, self, self);
			}
		}
    }
}
