// Coordinates the different parts of the piston platform-style items.

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
// In "fix mode", we're moving any misplaced pistons back into place.
// In that case up()/dn() redirect to fix().
g_fix <- false;

START_SND <- "";
STOP_SND <- "";

enable_motion_trig <- null;
fizz_up <- {player=null, obj=null, on=null, allowed=false};
fizz_dn <- {player=null, obj=null, on=null, allowed=false};
fizz_eager <- false;
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
		// No change. run fix() anyway.
		return fix();
	}
	g_fix = false;

	if (g_cur_moving == -1) {
		start();
	}
	door_pos = null;
	if (old_pos < new_pos) {
		enable_fizz(fizz_up);
		disable_fizz(fizz_dn);
		_up();
	} else if (old_pos > new_pos) {
		enable_fizz(fizz_dn);
		disable_fizz(fizz_up);
		_dn();
	}
}

function enable_fizz(fizz) {
	if (fizz.obj || fizz.player) {
		if (fizz_eager) {
			fizz.on = true;
			if (fizz.obj) {
				EntFireByHandle(fizz.obj, "Enable", "", 0, self, self);
			}
			if (fizz.player) {
				EntFireByHandle(fizz.player, "Enable", "", 0, self, self);
			}
		} else {
			fizz.allowed = true;
		}
	}
}

function disable_fizz(fizz) {
	if (fizz.obj || fizz.player) {
		fizz.allowed = false;
		if (fizz.on) {
			fizz.on = false;
			if (fizz.obj) {
				EntFireByHandle(fizz.obj, "Disable", "", 0, self, self);
			}
			if (fizz.player) {
				EntFireByHandle(fizz.player, "Disable", "", 0, self, self);
			}
		}
	}
}

function start() {
	if(START_SND) {
		self.EmitSound(START_SND);
	}
	if (self.GetClassname() == "ambient_generic") {
		EntFireByHandle(self, "PlaySound", "" 0.0, self, self);
	} else if (self.GetClassname() == "func_rotating") { // Alt technique
		EntFireByHandle(self, "Start", "", 0.0, self, self);
	} 
	if (enable_motion_trig != null) {
		EntFireByHandle(enable_motion_trig, "Enable", "", 0, self, self);
		EntFireByHandle(enable_motion_trig, "Disable", "", 0.1, self, self);
	}
}

// These two funcs find the first platform in their direction that's wrong,
// and trigger it.
// The pistons then trigger them again when they finish, so we loop until done.
function _up() {
	if (g_fix) {return fix(); }
	for(local i=1; i<=g_target_pos; i++) {
		if (g_positions[i] != POS_UP) {
			g_positions[i] = POS_MOVING;
			EntFireByHandle(g_pistons[i], g_inverted[i] ? "Close" : "Open", "", 0, self, self);
			g_cur_moving = i;
			door_pos = g_pistons[i].GetOrigin();
			crush_count = 0;
			return;
		}
	}
	// Finished.
	fix();
	disable_fizz(fizz_up);
}

function _dn() {
	if (g_fix) { return fix(); }
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
	door_pos = null;
	disable_fizz(fizz_dn);
	fix();
}

// After we think we're in position, check all pistons to confirm,
// and move any that aren't back. 
function fix() {
	foreach(i, ent in g_pistons) {
		if (i <= g_target_pos) { // Should be up
			if (g_positions[i] != POS_UP) {
				EntFireByHandle(g_pistons[i], g_inverted[i] ? "Close" : "Open", "", 0, self, self);
				if (g_cur_moving == -1) {
					start();
				}
				g_cur_moving = i;
				g_fix = true;
				return;
			}
		} else {
			if (g_positions[i] != POS_DN) {
				EntFireByHandle(g_pistons[i], g_inverted[i] ? "Open" : "Close", "", 0, self, self);
				if (g_cur_moving == -1) {
					start();
				}
				g_cur_moving = i;
				g_fix = true;
				return;
			}
		}
	}
	if (g_cur_moving != -1) {
		// We were moving, stop sounds.
		if (STOP_SND) {
			self.EmitSound(STOP_SND);
		}
		if (self.GetClassname() == "ambient_generic") {
			EntFireByHandle(self, "StopSound", "" 0.0, self, self);
		} else if (self.GetClassname() == "func_rotating") { // Looping sound
			EntFireByHandle(self, "Stop", "", 0.00, self, self);
		} 
	}
	g_cur_moving = -1;
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
    if ((fizz_dn.allowed || fizz_up.allowed) && crush_count < 25 && g_cur_moving != -1 && door_pos != null) {
		local new_pos = g_pistons[g_cur_moving].GetOrigin();
		if ((new_pos - door_pos).LengthSqr() < 1) {
			crush_count++;
			fizz_crush(fizz_dn);
			fizz_crush(fizz_up);
		} else {
			crush_count = 0;
		}
		door_pos = new_pos;
   		return 0.1;
    }
    return g_cur_moving != -1 ? 0.1 : 0.25;
}

function fizz_crush(fizz) {
	if (!fizz.allowed) {
		return;
	}
	fizz.on = true;
	if (crush_count == 2 && fizz.obj) {
		// Stuck, fizzle objects.
		EntFireByHandle(fizz.obj, "Enable", "", 0, self, self);
	}
	if (crush_count == 25 && fizz.player) {
		// Wait a bit longer for players, so there's time for objects
		// to dissolve first. It takes 2s to fizzle a cube.
		EntFireByHandle(fizz.player, "Enable", "", 0, self, self);
	}
}
