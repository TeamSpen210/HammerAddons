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

// Nodes with outputs that need to be delayed.
class Output {
	time = 0.0; // Delay after anim start before it should fire.
	target = null; // The entity to fire inputs at.
	input = null; // The input to use.
	is_tv = false;
	constructor (_time, ent_name, input_name) {
		time = _time;
		target = Entities.FindByName(null, ent_name);
		// Special input name - if used, fire a skin input instead.
		if (input_name == "<SKIN>") {
			is_tv = true;
			input = "Skin"
		} else {
			is_tv = false;
			input = input_name;
		}
	}
	function _tostring() { return tostring() }
	function tostring() {
	    return "<Output @" + time + ", targ: \""+target.GetName() + "\", input=" + input + ", tv=" + is_tv + ">";
	}
}

// A specific animation/route.
class Anim {
	name = ""; // Name of the animation.
	cargo_type = null; // Cube type it spawns.
	req_spawn = false; // If the dropper needs a new cube.
	pass_io = []; // Outputs to delay-fire.
	duration = 0.0; // Length of animation.

	opt_fast_reflection = false;
	opt_no_rtt_shadow = false;
	opt_no_rec_projtex = false;
	constructor (
		anim_name, time, cube_type, pass_io_lst, 
		fast_ref, no_rtt_shadow, no_rec_projtex,
	) {
		name = anim_name;
		duration = time;
		cargo_type = cube_type;
		req_spawn = false;
		pass_io = pass_io_lst;

		opt_fast_reflection = fast_ref;
		opt_no_rtt_shadow = no_rtt_shadow;
		opt_no_rec_projtex = no_rec_projtex;
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
	opt_skin = 0;
	opt_fast_reflection = false;
	opt_no_rtt_shadow = true;
	opt_no_rec_projtex = false;
	constructor (time, mov, vis) {
		reuse_time = time;
		mover = mov;
		visual = vis;
		opt_skin = 0;
		opt_fast_reflection = false;
		opt_no_rtt_shadow = true;
		opt_no_rec_projtex = false;
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

// Cache the keyvalues we previously set, to reduce IO spam. For each bool option,
// this has the attribute, and on/off input.
PROP_INPUTS <- [
	{attr="opt_fast_reflection", enable="EnableDrawInFastReflection", disable="DisableDrawInFastReflection"},
	{attr="opt_no_rtt_shadow", enable="EnableShadow", disable="DisableShasow"},
	{attr="opt_no_rec_projtex", enable="EnableReceivingFlashlight", disable="DisableReceivingFlashlight"},
]

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
function anim(
	anim_name, time, type, pass_io_lst,
	fast_ref, no_rtt_shadow, no_rec_projtex,
) {
	local ani = Anim(
		anim_name, time, type, pass_io_lst,
		fast_ref, no_rtt_shadow, no_rec_projtex
	);
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
		// printl("Vactube ent count: " + (::vactube_objs.len() * 2).tostring());
	} else {
		// Teleport to the right position.
		cargo.mover.SetAbsOrigin(self.GetOrigin());
	}

	cargo.visual.SetModel(cargo_type.model);
	if (cargo.opt_skin != cargo_type.skin) {
		EntFireByHandle(cargo.visual, "Skin", cargo_type.skin.tostring(), 0, self, self);
		cargo.opt_skin = cargo_type.skin;
	}
    EntFireByHandle(cargo.visual, "EnableDraw", "", 0, self, self);
    EntFireByHandle(cargo.visual, "SetLocalOrigin", cargo_type.localpos, 0, self, self);
    EntFireByHandle(cargo.mover, "SetAnimation", anim.name, 0, self, self);
    EntFireByHandle(cargo.visual, "DisableDraw", "", anim.duration, self, self);
    cargo.reuse_time = cur_time + anim.duration + 0.1; // Make sure enable/disable inputs don't get mixed up.

    // Apply these options, with a cache to avoid re-firing.
    foreach (opt in PROP_INPUTS) {
		local desired = anim[opt.attr]
    	if (cargo[opt.attr] != desired) {
			EntFireByHandle(cargo.visual, desired ? opt.enable : opt.disable, "", 0, self, self);
			cargo[opt.attr] = desired;
    	}
    }

    foreach (output in anim.pass_io) {
		// Skin inputs for TVs need the cargo skin.
		local param;
		if (output.is_tv) {
			param = cargo_type.tv_skin.tostring();
		} else {
			param = "";
		}
		// Do not pass an !activator here. The cargo props are shared, so 
		// users shouldn't be doing anything to them. In particular, 
		// OnPass -> kill outputs may be present which are not useful.
		EntFireByHandle(output.target, output.input, param, output.time, null, null);
    }
}
