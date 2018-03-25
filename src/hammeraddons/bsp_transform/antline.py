"""Add "Indicator Name" keys to Portal 2 entities.

This generates env_texturetoggle entities which do the right thing.
If the one of the target entities is a prop_indicator_panel, it also 
toggles that. 
"""

# Entities with the keyvalue -> on, off output names.
IND_ENTS = {
    'prop_button': ('OnPressed', 'OnButtonReset'),
    'prop_under_button': ('OnPressed', 'OnButtonReset'),
    
    'prop_floor_button': ('OnPressed', 'OnUnPressed'),
    'prop_floor_cube_button': ('OnPressed', 'OnUnPressed'),
    'prop_floor_ball_button': ('OnPressed', 'OnUnPressed'),
    'prop_under_floor_button': ('OnPressed', 'OnUnPressed'),
    
    'prop_laser_catcher': ('OnPowered', 'OnUnPowered'),
    'prop_laser_relay': ('OnPowered', 'OnUnPowered'),
    'point_laser_target': ('OnPowered', 'OnUnPowered'),
    
    'trigger_multiple': ('OnStartTouch', 'OnEndTouchAll'),
}


