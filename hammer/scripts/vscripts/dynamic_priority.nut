// Made for the dynamic priority stuff


::DYNPR_SCOPE <- Storage.CreateScope("DYNAMIC_PRIORITY");


function ChangeMode(modename) {

    local name = ""

    DYNPR_SCOPE.SetInt("current_mode", modename);

    switch (modename) {
        case 2:
            ActivateLow();
            ActivateMedium();
            name = "High";
            break;

        case 0:
            DeactivateLow();
            ActivateMedium();
            name = "Medium";
            break;

        case 1:
            DeactivateMedium();
            DeactivateLow();
            name = "Low";
            break;

        default:
            printl("No setting for " + modename + "!");
            printl("Reverting to default!");
            DYNPR_SCOPE.SetInt("current_mode", 0);
            ChangeMode(0)
            return;
    }

    printl("Changing mode to " + name + "!");
}

function DeactivateLow() {
    EntFire("light_dynpr_dynamic_0", "TurnOff", "", 0, null);
    EntFire("light_dynpr_static_0", "TurnOn", "", 0, null);
}

function DeactivateMedium() {
    EntFire("light_dynpr_dynamic_1", "TurnOff", "", 0, null);
    EntFire("light_dynpr_static_1", "TurnOn", "", 0, null);
}


function ActivateLow() {
    EntFire("light_dynpr_dynamic_0", "TurnOn", "", 0, null);
    EntFire("light_dynpr_static_0", "TurnOff", "", 0, null);
}

function ActivateMedium() {
    EntFire("light_dynpr_dynamic_1", "TurnOn", "", 0, null);
    EntFire("light_dynpr_static_1", "TurnOff", "", 0, null);
}



function LoadFromMemory() {

    local cur_mode = 0;

    // 0 Medium
    // 1 Low
    // 2 High

    // This is mixed up as GetInt seems to return 0 if value wasn't initialized

    try {
        cur_mode = DYNPR_SCOPE.GetInt("current_mode");
    } catch (err) { 
        DYNPR_SCOPE.SetInt("current_mode", 2);
        cur_mode = 2;
    }

    ChangeMode(cur_mode);
}