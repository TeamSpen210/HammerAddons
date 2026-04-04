"""Calculates the UVs for a vgui_movie_display to tile with neighbours."""
from enum import Enum
import itertools

from srctools import Entity, FrozenVec, Matrix, conv_float, Vec, conv_int, conv_bool, Output
from srctools.logger import get_logger
import attrs

from hammeraddons.bsp_transform import Context, ConfOpt, Config, trans
from hammeraddons.bsp_transform.common import ent_description, strip_cust_keys


class Direction(Enum):
    """Attachment directions."""
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


LOGGER = get_logger(__name__)
OFFSETS = [
    (Direction.UP, 0.5, 1),
    (Direction.DOWN, 0.5, 0),
    (Direction.LEFT, 0, 0.5),
    (Direction.RIGHT, 1, 0.5),
]
LINKS = [
    (Direction.UP, Direction.DOWN),
    (Direction.RIGHT, Direction.LEFT),
]

CONFIG = Config('comp_movie_fitter', 1, [
    CONF_THRESHOLD := ConfOpt.floating(
        'neighbour_range', 32.0,
        """The midpoint of two edges must be at least this close to be considered joined."""
    ),
])


@attrs.define(eq=False, repr=False)
class Screen:
    """Each screen we can link."""
    ent: Entity
    links: dict[Direction, 'Screen | None']
    pos: dict[Direction, FrozenVec]
    # If set, one of the neighbours was duplicated or otherwise invalid. Compile error later if we
    # try to fit, but only warn if its in a group we never get to.
    invalid: bool = False

    def __repr__(self) -> str:
        return f'<Screen {ent_description(self.ent)}>'


@trans('comp_movie_fitter')
def comp_movie_fitter(ctx: Context) -> None:
    """Calculate UVs for a vgui_movie_display."""
    if not ctx.vmf.by_class['comp_movie_fitter']:
        return  # Don't do the expensive neighbour calculations.

    screens: dict[Entity, Screen] = {}
    for ent in ctx.vmf.by_class['vgui_movie_display']:
        orient = Matrix.from_angstr(ent['angles'])
        origin = Vec.from_str(ent['origin'])
        width = conv_float(ent['width'])
        height = conv_float(ent['height'])
        screens[ent] = Screen(
            ent,
            links=dict.fromkeys(Direction, None),
            pos={direct: (FrozenVec(0, y * width, z * height) @ orient + origin) for direct, y, z in OFFSETS},
        )

    LOGGER.info('{} screens found, linking neighbours...', len(screens))
    threshold = CONFIG.get(CONF_THRESHOLD) ** 2.0

    for screen_a, screen_b in itertools.product(screens.values(), screens.values()):
        if screen_a is screen_b:
            continue
        # We just compare one direction, the other will be done later in the loop.
        for dir_a, dir_b in LINKS:
            if (screen_a.pos[dir_a] - screen_b.pos[dir_b]).mag_sq() > threshold:
                continue
            if screen_a.links[dir_a] is None and screen_b.links[dir_b] is None:
                screen_a.links[dir_a] = screen_b
                screen_b.links[dir_b] = screen_a
            else:
                # Warn and invalidate if either are linked to something else.
                bad = screen_a.links[dir_a]
                if bad is not None and bad is not screen_b:
                    LOGGER.warning(
                        'Three screens overlap edges:\n{}\n{}\n{}',
                        ent_description(screen_a.ent),
                        ent_description(screen_b.ent),
                        ent_description(bad.ent)
                    )
                    screen_a.invalid = screen_b.invalid = bad.invalid = True
                bad = screen_b.links[dir_b]
                if bad is not None and bad is not screen_a:
                    LOGGER.warning(
                        'Three screens overlap edges:\n{}\n{}\n{}',
                        ent_description(screen_a.ent),
                        ent_description(screen_b.ent),
                        ent_description(bad.ent)
                    )
                    screen_a.invalid = screen_b.invalid = bad.invalid = True
    LOGGER.info('Linked screens.')
    for fitter in ctx.vmf.by_class['comp_movie_fitter']:
        for ent in ctx.vmf.search(fitter['target']):
            try:
                lower_left = screens[ent]
                break
            except KeyError:
                pass
        else:
            raise ValueError(
                f'Fitter {ent_description(fitter)} could not '
                f'find screen "{fitter['target']}"!'
            )

        screen_pos = {lower_left: (0, 0)}
        queue = [(lower_left, 0, 0)]
        min_x = min_y = max_x = max_y = 0
        while queue:
            screen, x, y = queue.pop()
            for direct in Direction:
                if (neighbour := screen.links[direct]) is None:
                    continue
                if neighbour.invalid:
                    raise ValueError(
                        'Cannot fit screen with inconsistent overlaps: '
                        f'{ent_description(neighbour.ent)} '
                        'See above for invalid overlapping.'
                    )
                off_x, off_y = direct.value
                new_x, new_y = x + off_x, y + off_y
                min_x = min(min_x, new_x)
                min_y = min(min_y, new_y)
                max_x = max(max_x, new_x)
                max_y = max(max_y, new_y)
                try:
                    cur_x, cur_y = screen_pos[neighbour]
                except KeyError:
                    screen_pos[neighbour] = new_x, new_y  # Add it
                    queue.append((neighbour, new_x, new_y))
                else:
                    if cur_x != new_x or cur_y != new_y:
                        raise ValueError(
                            f'Inconsistent position for screen {ent_description(neighbour.ent)}. '
                            "Screens which don't fit a uniform grid are not supported."
                        )
        LOGGER.info(
            'Fitter {} has {} screens in a {} x {} grid',
            ent_description(fitter), len(screen_pos),
            max_x - min_x + 1, max_y - min_y + 1,
        )
        off_x = off_y = 0
        tile_width = conv_int(fitter['tile_width'], -1)
        tile_height = conv_int(fitter['tile_height'], -1)
        tile_once = conv_bool(fitter['tile_once'])
        if tile_width <= 0:
            tile_width = max_x + 1
            off_x = min_x
        if tile_height <= 0:
            tile_height = max_y + 1
            off_y = min_y

        match fitter['mode'].casefold():
            case 'spawn':
                fitter['classname'] = 'logic_auto'
                fitter['targetname'] = ''
                output = 'OnMapSpawn'
            case 'trigger':
                fitter['classname'] = 'logic_relay'
                output = 'OnTrigger'
            case _:
                raise ValueError(f'Invalid comp_movie_fitter mode "{fitter['mode']}"!')
        strip_cust_keys(fitter)
        LOGGER.debug('Fitting to {}x{}, off={} {}', tile_width, tile_height, off_x, off_y)

        for screen, (x, y) in screen_pos.items():
            if tile_once and (x < 0 or x > tile_width or y < 0 or y > tile_width):
                continue
            min_u = ((x - off_x) / tile_width) % 1.0
            max_u = ((x + 1 - off_x) / tile_width) % 1.0
            min_v = ((y - off_y) / tile_height) % 1
            max_v = ((y + 1 - off_y) / tile_height) % 1.0
            if max_u == 0:
                max_u = 1.0
            if max_v == 0:
                max_v = 1.0
            LOGGER.debug('Fitting: {}, {} = ({} {}) / ({} {})', x, y, min_u, max_u, min_v, max_v)
            name = screen.ent['targetname']
            fitter.add_out(
                Output(output, name, 'SetUseCustomUVs', '1'),
                Output(output, name, 'SetUMin', min_u),
                Output(output, name, 'SetUMax', max_u),
                Output(output, name, 'SetVMin', min_v),
                Output(output, name, 'SetVMax', max_v),
            )
