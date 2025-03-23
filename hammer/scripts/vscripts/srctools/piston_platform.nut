// Coordinates the different parts of the piston platform-style items.

g_target_pos <- 0;  // Position we want to be at.
// Layout:
// pos, pist
// 4    _ _
//       4
// 3    _4_
//       3
// 2    _3_
//       2 
// 1    _2_  ___
//       1    |
// 0 ____1____|_

// 1-max = ents, set by compiler.
g_pistons <- {};
// For each piston, if open/closed is swapped.
g_inverted <- {};
// positions - where the door is right now.
POS_UP <- 1;
POS_DN <- -1;
POS_MOVING <- 0;
g_positions <- {}; // Initialised by compiler.
// Currently moving piston, or -1 if stationary.
g_cur_moving <- -1;
// Maximum piston index.
MAX_IND <- 4;
// Position we're trying to move to.
g_target_pos <- 0;

START_SND <- "";
STOP_SND <- "";

enable_motion_trig <- null;
dn_fizz_ents <- [];
dn_fizz_on <- false;
dn_fizz_allowed <- false;
door_pos <- null;
crush_count <- 0;
snd_btm_pos <- self.GetOrigin();
snd_top_ent <- null;


function Precache() {
	if(START_SND) self.PrecacheSoundScript(START_SND);
	if(STOP_SND)  self.PrecacheSoundScript(STOP_SND);
}

function moveto(new_pos) {
	local old_pos = g_target_pos;
	g_target_pos = new_pos;
	
	printl("Moving: " + old_pos + " -> " + new_pos);
	
	if (old_pos == new_pos) {
		return; // No change.
	}
	
	if (g_cur_moving == -1) {
		if(START_SND) {
			self.EmitSound(START_SND);
		}
		if (self.GetClassname() == "func_rotating") { // Looping sound
			EntFireByHandle(self, "Start", "", 0.00, self, self);
		}
		if (enable_motion_trig != null) {
			EntFireByHandle(enable_motion_trig, "Enable", "", 0, self, self);
			EntFireByHandle(enable_motion_trig, "Disable", "", 0.1, self, self);
		}
	}
	
	if (old_pos < new_pos) {
		door_pos = null;
		if (dn_fizz_ents.len() > 0) {
			dn_fizz_allowed = false;
			if (dn_fizz_on) {
				dn_fizz_on = false;
				foreach (fizz in dn_fizz_ents) {
					EntFireByHandle(fizz, "Disable", "", 0, self, self);
				}
			}
		}
		_up();
	} else if (old_pos > new_pos) {
		_dn();
		if (dn_fizz_ents.len() > 0) {
			dn_fizz_allowed <- true;
		}
	}
}

// These two funcs find the first platform in their direction that's wrong,
// and trigger it.
// The pistons then trigger them again when they finish, so we loop until done.
function _up(index=null) {
	for(local i=1; i<=g_target_pos; i++) {
		if (g_positions[i] != POS_UP) {
			g_positions[i] = POS_MOVING;
			EntFireByHandle(g_pistons[i], g_inverted[i] ? "Close" : "Open", "", 0, self, self);
			g_cur_moving = i;
			return;
		}
	}
	// Finished.
	g_cur_moving = -1;
	if (STOP_SND) {
		self.EmitSound(STOP_SND);
	}
	if (self.GetClassname() == "func_rotating") { // Looping sound
		EntFireByHandle(self, "Stop", "", 0.00, self, self);
	}
}

function _dn(index=null) {
	// Do not include g_pistons[pos].
	for(local i=MAX_IND; i>g_target_pos; i--) {
		if (g_positions[i] != POS_DN) {
			g_positions[i] = POS_MOVING;
			EntFireByHandle(g_pistons[i], g_inverted[i] ? "Open" : "Close", "", 0, self, self);
			g_cur_moving = i;
			door_pos = g_pistons[i].GetOrigin();
			crush_count = 0;
			return;
		}
	}
	// Finished.
	g_cur_moving = -1;
	if (STOP_SND) {
		self.EmitSound(STOP_SND);
	}
	if (self.GetClassname() == "func_rotating") { // Looping sound.
		EntFireByHandle(self, "Stop", "", 0.00, self, self);
	}
	if (dn_fizz_on) {
		dn_fizz_on = false;
		dn_fizz_allowed = false;
		door_pos = null;
		foreach (fizz in dn_fizz_ents) {
			EntFireByHandle(fizz, "Disable", "", 0, self, self);
		}
	}
}

function Think() {
	if (g_cur_moving != -1 && snd_top_ent != null) {
		// Update position.
		local sum = snd_btm_pos + snd_top_ent.GetOrigin();
		sum *= 0.5;
		self.SetOrigin(sum);
	}

	// Used by pistons that can fizzle objects below them.
	// If it gets stuck (stops moving), activate.
	// Lotsa checks here.
	// Only run if:
	// * allowed to.
	// * Not already on
	// * Currently moving
	// * We have a valid previous position
	// It has to trigger twice consecuatively.
    if (dn_fizz_allowed && !dn_fizz_on && g_cur_moving != -1 && door_pos != null) {
		local new_pos = g_pistons[g_cur_moving].GetOrigin();
		if ((new_pos - door_pos).LengthSqr() < 1) {
			crush_count++;
			if (crush_count > 2) {
				// Stuck...
				dn_fizz_on = true;
				foreach (fizz in dn_fizz_ents) {
					EntFireByHandle(fizz, "Enable", "", 0, self, self);
				}
			}
		} else {
			crush_count = 0;
		}
		door_pos = new_pos;
   		return 0.05;
    }
    return g_cur_moving != -1 ? 0.1 : 0.25;
}
