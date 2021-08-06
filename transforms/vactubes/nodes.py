"""Implements the various curve types for vactubes."""
import math
import bisect
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Iterator, Iterable, Dict, Optional, List, ClassVar

import srctools.logger
from srctools import Vec, Entity, VMF, Output, conv_bool, Angle, Matrix, lerp


LOGGER = srctools.logger.get_logger(__name__)
PASS_OUT = 'User4'  # User output to use to trigger onPass etc outputs.
CUBE_MODEL = 'models/props/metal_box.mdl'
# A bit smaller than the size of the model, so we can determine when to
# start/stop showing ents on the screen.
SCANNER_LENGTH = 80

# A list of the models the geocable transform generated.
SPLINES: List['Spline'] = []


class DestType(Enum):
    """The position of the output from a node.."""
    PRIM = PRIMARY = 'primary'
    SEC = SECONDARY = 'secondary'
    TER = TERTIARY = 'tertiary'
    OUT = PRIMARY

    @property
    def manual_targ(self) -> str:
        """The name of the manual target keyvalue."""
        v = self.value
        if v == 'primary':
            return 'target'
        elif v == 'secondary':
            return 'target_sec'
        else:
            return 'target_tri'


def curve_point(radius: float, t: float) -> Tuple[float, float]:
    """Compute the offset along a curve, in XY.

    The curve goes from the [1, 0] direction to the [0, 1] direction.
    It starts at 0,0 and ends at 1, 1.
    """
    ang = math.pi * t / 2.0
    return radius * math.sin(ang), radius - radius * math.cos(ang)


class Node(ABC):
    """A node is a junction or curve in the track.

    This also defines the behaviour of the point.

    This has up to 3 output connections, depending on type.
    """
    # The name of the output fired when cubes pass this.
    pass_out_name: ClassVar[str] = 'onpass'
    # If true, keep the entity around regardless. User4 is reserved for the pass output.
    keep_ent: ClassVar[bool] = False
    # The outputs the item can use.
    out_types: ClassVar[Iterable[DestType]] = ()

    def __init__(self, ent: Entity) -> None:
        self.origin = Vec.from_str(ent['origin'])
        self.matrix = Matrix.from_angle(Angle.from_str(ent['angles']))
        self.ent = ent

        self.has_input = False  # We verify every node has an input if used.
        # DestType -> output.
        self.outputs: Dict[DestType, Optional[Node]] = dict.fromkeys(self.out_types, None)
        # Outputs fired when cubes reach this point.
        pass_outputs = [
            out for out in ent.outputs
            if out.output.casefold() == self.pass_out_name
        ]
        self.has_pass = bool(pass_outputs)
        if self.has_pass:
            for out in pass_outputs:
                out.output = 'On' + PASS_OUT
            if ent['classname'].startswith('comp_'):
                # Remove the extra keyvalues we use.
                ent.keys = {
                    'classname': 'info_target',
                    'targetname': ent['targetname'],
                    'origin': ent['origin'],
                    'angles': ent['angles'],
                }
            ent.make_unique('_vac_node')
        elif not self.keep_ent:
            ent.remove()

    @property
    def name(self) -> str:
        """The name of the entity."""
        return self.ent['targetname']

    def __repr__(self) -> str:
        return '<{} "{}" @ {}, {}>'.format(
            self.__class__.__name__,
            self.name,
            self.origin,
            self.matrix.to_angle(),
        )

    @abstractmethod
    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        """Return the length of the track for this node.

        If zero, it's just a point position.
        """
        raise NotImplementedError(self)

    @abstractmethod
    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the position of the track.

        T=0 means the start, T=1 is the end.
        """
        raise NotImplementedError(self)

    @abstractmethod
    def input_norm(self) -> Vec:
        """Return the flow direction at the input point."""

    @abstractmethod
    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the output node."""

    def tv_code(self, item_speed: float) -> str:
        """If a scanner, return the VScript for the scanner prop name and shutoff delay."""
        return 'null, 0.0'


def parse(vmf: VMF) -> Iterator[Node]:
    """Parse out all the vactube nodes from the VMF."""
    for ent in vmf.by_class['comp_vactube_start']:
        yield Spawner(ent)

    for ent in vmf.by_class['comp_vactube_end']:
        cube_radius = srctools.conv_float(ent['radius'])
        if cube_radius > 0:
            yield Dropper.parse(vmf, ent, cube_radius)
        else:
            yield Destroyer(ent)

    for ent in vmf.by_class['comp_vactube_junction']:
        is_reversed = srctools.conv_int(ent['skin']) == 1

        orig_mdl = model = ent['model'].casefold()
        del ent['model']
        if not model.startswith('models/editor/vactubes/'):
            raise ValueError(
                f'Model "{orig_mdl}" is not a valid vactube '
                f'junction type (at {ent["origin"]})'
            )
        model = model[23:]
        if model == "straight.mdl":
            yield Straight(ent)
        elif model == "diag_curve.mdl":
            yield DiagCurve(ent, is_reversed, False)
        elif model == "diag_curve_mirror.mdl":
            yield DiagCurve(ent, is_reversed, True)
        elif model[:6] == "curve_" and model[-4:] == ".mdl":
            # Each curve has a 64 units wider radius. Parse the model,
            # so users can add larger curves if they want.
            ind = int(model[6:-4])
            yield Curve(ent, 64.0 * ind, is_reversed)
        elif model == "splitter_straight.mdl":
            yield Splitter(ent, True)
        elif model == "splitter_sides.mdl":
            yield Splitter(ent, False)
        elif model == "splitter_triple.mdl":
            yield CrossSplitter(ent)
        else:
            raise ValueError(
                f'Model "{orig_mdl}" is not a valid vactube '
                f'junction type (at {ent["origin"]})'
            )
    yield from SPLINES


class Spawner(Node):
    """The start point of the track."""
    pass_out_name = 'onspawned'
    keep_ent = True
    out_types = [DestType.PRIMARY]

    def __init__(self, ent: Entity) -> None:
        super().__init__(ent)
        self.group = ent['group'].casefold().strip()
        self.speed = srctools.conv_float(ent['speed'], 800.0)
        self.is_auto = srctools.conv_bool(ent['timer'])
        self.seed = ent['seed']
        if not self.seed:
            # Generate a new seed, and notify the user so they can copy it down
            # if they want to use it themselves.
            self.seed = format(random.getrandbits(64), '08X')
        LOGGER.info('Spawner "{}" using random seed "{}"', self.name, self.seed)

        if self.is_auto:
            self.time_min = srctools.conv_float(ent['time_min'], 0.5)
            self.time_max = srctools.conv_float(ent['time_max'], 1.0)
        else:
            self.time_min = self.time_max = 0.0

        # Strip these keyvalues.
        del ent['speed'], ent['timer'], ent['time_min'], ent['time_max']

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        assert dest is DestType.PRIMARY, self
        return self.origin

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        assert dest is DestType.PRIMARY, self
        return 0.0

    def input_norm(self) -> Vec:
        raise AssertionError(self)

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        assert dest is DestType.PRIMARY
        return Vec(x=1) @ self.matrix


class Destroyer(Node):
    """The end of the track."""
    pass_out_name = 'oncubearrived'
    out_types: ClassVar[Iterable[DestType]] = []

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        assert dest is DestType.PRIMARY
        return self.origin

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        assert dest is DestType.PRIMARY
        return 0.0

    def input_norm(self) -> Vec:
        return Vec(x=-1) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Destroyers have no outputs."""
        raise AssertionError(self)


class Spline(Node):
    """Special node matching a model generated by the geocable transform."""
    out_types: ClassVar[Iterable[DestType]] = [DestType.PRIMARY]
    _vmf = VMF()

    def __init__(self, origin: Vec, points: List[Vec]) -> None:
        # Create a dummy entity, we don't really need it.
        super().__init__(self._vmf.create_ent('comp_vactube_spline', origin=origin))
        assert len(points) >= 2, 'Not enough points!'
        self.start_norm = (points[1] - points[0]).norm()
        self.end_norm = (points[-1] - points[-2]).norm()

        self.points = points
        length = 0.0
        # Accumulate the lengths, so we can interpolate.
        self.lengths = []
        last_pos = points[0]
        for pos in points:
            length += (pos - last_pos).mag()
            self.lengths.append(length)
            last_pos = pos
        self.length = length

    def path_len(self, dest: DestType = DestType.PRIMARY) -> float:
        """Return the total length of this path."""
        assert dest is DestType.PRIMARY
        return self.length

    def vec_point(self, t: float, dest: DestType = DestType.PRIMARY) -> Vec:
        """Given an interpolation point between 0-1, find the location at that point."""
        assert dest is DestType.PRIMARY
        dist = self.length * t
        i = bisect.bisect_left(self.lengths, dist)
        pos1, len1 = self.points[i], self.lengths[i]
        try:
            pos2, len2 = self.points[i+1], self.lengths[i+1]
        except IndexError: # End of list.
            return self.origin + pos1
        return self.origin + Vec(
            lerp(dist, len1, len2, pos1.x, pos2.x),
            lerp(dist, len1, len2, pos1.y, pos2.y),
            lerp(dist, len1, len2, pos1.z, pos2.z),
        )

    def input_norm(self) -> Vec:
        """Return the direction of the input side."""
        return self.start_norm.copy()

    def output_norm(self, dest: DestType = DestType.PRIMARY) -> Vec:
        """return the direction of the output side."""
        assert dest is DestType.PRIMARY
        return self.end_norm.copy()


class Dropper(Destroyer):
    """The endpoint which is linked to/controls a dropper."""
    keep_ent = True

    def __init__(self, ent: Entity, temp: Entity, cube: Entity) -> None:
        super().__init__(ent)
        self.template = temp
        self.cube = cube

    @classmethod
    def parse(cls, vmf: VMF, ent: Entity, radius: float) -> 'Dropper':
        """Scan the map applying dropper tweaks, then create the Dropper object."""
        filter_name = ent['filtername']
        template_name = ent['template']

        for cube_filter in vmf.search(filter_name):
            break
        else:
            raise ValueError(f'No filter "{filter_name}" for dropper at {ent["origin"]}!')

        for template in vmf.search(template_name):
            break
        else:
            raise ValueError(f'No template "{template_name}" for dropper at {ent["origin"]}!')

        best_cube = None
        best_dist = math.inf
        radius **= 2
        ref_pos = Vec.from_str(cube_filter['origin'])
        for cube in vmf.by_class['prop_weighted_cube'] | vmf.by_class['prop_monster_box']:
            dist = (Vec.from_str(cube['origin']) - ref_pos).mag_sq()
            if dist > radius or dist > best_dist:
                continue
            best_dist = dist
            best_cube = cube
        if best_cube is None:
            LOGGER.warning('Cube dropper at {} has no cube. Generating standard one...', ref_pos)
            best_cube = vmf.create_ent(
                'prop_weighted_cube',
                angles='0 0 0',
                newskins='1',
                skintype='0',
                cubetype='0',
                skin='0',
                paintpower='4',
                model=CUBE_MODEL,
            )

        # Now adjust the cube for dropper use.
        best_cube.make_unique('dropper_cube')
        best_cube['origin'] = ent['origin']
        # Only regular cubes can disable funnelling, but frankenturrets
        # require being in box form.
        if best_cube['classname'] == 'prop_monster_box':
            best_cube['startasbox'] = '1'
        else:
            best_cube['allowfunnel'] = '0'

        # Copy the cube name to filter and dropper.
        cube_filter['filtername'] = best_cube['targetname']

        for i in range(1, 10):
            if not template[f'Template{i:02}']:
                template[f'Template{i:02}'] = best_cube['targetname']
            break
        else:
            raise ValueError(f'No spare slots for template "{template_name}"!')

        # Add fizzle outputs if enabled.
        if srctools.conv_bool(ent['autorespawn']):
            best_cube.outputs += [
                out for out in ent.outputs
                if out.output.casefold() == 'onfizzled'
            ]
        ent.add_out(Output(Dropper.pass_out_name, template, 'ForceSpawn'))
        return Dropper(ent, template, best_cube)


class Curve(Node):
    """A simple corner node."""
    out_types = [DestType.PRIMARY]

    def __init__(self, ent: Entity, radius: float, reversed: bool) -> None:
        super().__init__(ent)
        self.radius = radius
        self.reversed = reversed

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        """Return the length of the curve."""
        assert dest is DestType.PRIMARY
        # πD / 4
        return math.pi * (2.0 / 4.0) * self.radius

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the position along the curve."""
        assert dest is DestType.PRIMARY

        if self.reversed:
            t = 1.0 - t

        x, y = curve_point(self.radius, t)
        return self.origin + Vec(0, x, -y) @ self.matrix

    def input_norm(self) -> Vec:
        if self.reversed:
            return Vec(z=1) @ self.matrix
        else:
            return Vec(y=1) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the end of this curve type."""
        assert dest is DestType.PRIMARY
        if self.reversed:
            return Vec(y=-1) @ self.matrix
        else:
            return Vec(z=-1) @ self.matrix


class DiagCurve(Node):
    """A 45 degree curve.

    This has a lot of precise constants to match a specific model.
    """
    out_types = [DestType.PRIMARY]
    CURVE_LEN = 155.0  # Manually integrated the curve function.
    STRAIGHT_LEN = 56.0
    TOTAL_LEN = CURVE_LEN + STRAIGHT_LEN
    STRAIGHT_PERC = STRAIGHT_LEN / TOTAL_LEN

    def __init__(self, ent: Entity, reversed: bool, flipped: bool) -> None:
        super().__init__(ent)
        self.reversed = reversed
        # If flipped, we just want to flip the sign of the Y coord.
        self.y = -1.0 if flipped else 1.0

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        """Return the length of the curve."""
        assert dest is DestType.PRIMARY
        return self.TOTAL_LEN

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the position along the curve."""
        assert dest is DestType.PRIMARY
        if self.reversed:
            t = 1.0 - t

        if t < self.STRAIGHT_PERC:
            x = 0.0
            y = lerp(t, 0.0, self.STRAIGHT_PERC, 128, 72)
        else:
            ang = lerp(t, self.STRAIGHT_PERC, 1.0, 0.0, math.pi/4)
            x = lerp(math.cos(ang), 1.0, math.cos(math.pi/4), 0, 64)
            y = lerp(math.sin(ang), 0.0, math.sin(math.pi/4), 72, -64)

        return Vec(x, self.y * y, 0) @ self.matrix + self.origin

    def input_norm(self) -> Vec:
        """Return the flow direction into the start of the curve."""
        if self.reversed:
            return Vec(x=-1, y=self.y).norm() @ self.matrix
        else:
            return Vec(y=-self.y) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the end of the curve."""
        assert dest is DestType.PRIMARY
        if self.reversed:
            return Vec(y=self.y) @ self.matrix
        else:
            return Vec(x=1, y=-self.y).norm() @ self.matrix


class Straight(Node):
    """A node pointing directly straight, for outputting or the like.

    This also checks for TV scanners being present nearby.
    """
    out_types = [DestType.PRIMARY]

    def __init__(self, ent: Entity):
        """Convert the entity to have the right logic."""
        self.scanner = None
        self.persist_tv = conv_bool(ent.keys.pop('persist_tv', False))

        pos = Vec.from_str(ent['origin'])
        for prop in ent.map.by_class['prop_dynamic']:
            if (Vec.from_str(prop['origin']) - pos).mag_sq() > 64**2:
                continue

            model = prop['model'].casefold().replace('\\', '/')
            # Allow spelling this correctly, if you're not Valve.
            if 'vacum_scanner_tv' in model or 'vacuum_scanner_tv' in model:
                self.scanner = prop
                prop.make_unique('_vac_scanner')
            elif 'vacum_scanner_motion' in model or 'vacuum_scanner_motion' in model:
                prop.make_unique('_vac_scanner')
                ent.add_out(Output(self.pass_out_name, prop, "SetAnimation", "scan01"))

        super(Straight, self).__init__(ent)

    def tv_code(self, item_speed: float) -> str:
        """If we have a scanner, return the associated data."""
        if self.scanner is None:
            return 'null, 0.0'
        # Compute the number of seconds it'll take to pass through the scanner.
        # Then divide by 2, since we want an offset from the middle of the scanner.
        # Limit to a minimum of 0.01, to ensure it doesn't round to zero.
        scan_time = max(0.01, SCANNER_LENGTH / item_speed / 2.0)

        # If we ask to persist the TV, it's only fired on entry.
        # So set this negative to signal that.
        if self.persist_tv:
            scan_time = -scan_time

        return f'"{self.scanner["targetname"]}", {scan_time:.3g}'

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        """Return the length of this node, which is always 32 units (arbitrarily)."""
        assert dest is DestType.PRIMARY
        return 32.0

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return points along the path inside this node."""
        assert dest is DestType.PRIMARY
        return self.origin + Vec(x=32.0*t - 16.0) @ self.matrix

    def input_norm(self) -> Vec:
        """Return the flow direction into the start of this node."""
        return Vec(x=1.0) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the end of this curve type."""
        assert dest is DestType.PRIMARY
        return Vec(x=1.0) @ self.matrix


class Splitter(Node):
    """A T-intersection that either randomly routes cubes or directs them to a dropper."""
    out_types = [DestType.PRIMARY, DestType.SECONDARY]

    def __init__(self, ent: Entity, straight: bool) -> None:
        """If straight is true, the primary dir goes forward."""
        super().__init__(ent)
        self.is_straight = straight

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        if self.is_straight and dest is DestType.PRIMARY:
            return 128.0
        else:
            return 64.0 / 4.0 * math.pi

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        assert dest is not DestType.TERTIARY
        x, y = curve_point(64.0, t)
        if dest is DestType.SECONDARY:
            return self.origin + Vec(y, x, 0) @ self.matrix
        elif self.is_straight:
            return self.origin + Vec(y=128*t) @ self.matrix
        else:
            return self.origin + Vec(-y, x, 0) @ self.matrix

    def input_norm(self) -> Vec:
        """Return the flow direction at the input side."""
        return Vec(y=1) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the end of this curve type."""
        if dest is DestType.SECONDARY:
            return Vec(x=1) @ self.matrix
        else:
            assert dest is DestType.PRIMARY
            if self.is_straight:
                return Vec(y=1) @ self.matrix
            else:
                return Vec(x=-1) @ self.matrix


class CrossSplitter(Node):
    """An X-intersection that either randomly routes cubes or directs them to a dropper.

    Primary is left, secondary is forward, tertiary is right.
    """
    out_types = [DestType.PRIMARY, DestType.SECONDARY, DestType.TERTIARY]

    def __init__(self, ent: Entity) -> None:
        """If straight is true, the primary dir goes forward."""
        super().__init__(ent)

    def path_len(self, dest: DestType=DestType.PRIMARY) -> float:
        if dest is DestType.SECONDARY:  # Straight through
            return 128.0
        else: # Either curve
            return 64.0 / 4.0 * math.pi

    def vec_point(self, t: float, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the position this far through the given curve."""
        x, y = curve_point(64.0, t)
        if dest is DestType.PRIMARY:
            return self.origin + Vec(y, x, 0) @ self.matrix
        elif dest is DestType.SECONDARY:
            return self.origin + Vec(y=128*t) @ self.matrix
        elif dest is DestType.TERTIARY:
            return self.origin + Vec(-y, x, 0) @ self.matrix
        else:
            raise AssertionError(dest)

    def input_norm(self) -> Vec:
        """Return the flow direction at the input side."""
        return Vec(y=1) @ self.matrix

    def output_norm(self, dest: DestType=DestType.PRIMARY) -> Vec:
        """Return the flow direction at the end of this curve type."""
        if dest is DestType.PRIMARY:
            return Vec(x=1) @ self.matrix
        elif dest is DestType.SECONDARY:
            return Vec(y=1) @ self.matrix
        elif dest is DestType.TERTIARY:
            return Vec(x=-1) @ self.matrix
        else:
            raise AssertionError(dest)
