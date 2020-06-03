import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Iterator

import srctools.logger
from srctools import Vec, Entity, VMF, Output


LOGGER = srctools.logger.get_logger(__name__)
PASS_OUT = 'User4'  # User output to use to trigger onPass etc outputs.
CUBE_MODEL = 'models/props/metal_box.mdl'


class ConnType(Enum):
    """The kind of point."""
    INPUT = 'input'
    OUTPUT = 'output'
    OUTPUT_SEC = 'output_sec'


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

    This is a input->output connection, and optionally an input->output_sec
    connection. For droppers the latter is the dropper.
    """
    # The name of the output fired when cubes pass this.
    pass_out_name: str = 'onpass'
    # If true, keep the entity around regardless. User4 is reserved for the pass output.
    keep_ent: bool = False

    def __init__(self, ent: Entity) -> None:
        self.origin = Vec.from_str(ent['origin'])
        self.angles = Vec.from_str(ent['angles'])
        self.ent = ent

        self.has_input = False  # We verify every node has an input if used.
        self.output = None  # type: Optional[Node]
        self.output_sec = None  # type: Optional[Node]
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

    @abstractmethod
    def path_len(self, sec: bool=False) -> float:
        """Return the length of the track for this node.

        If zero, it's just a point position.
        """
        raise NotImplementedError(self)

    @abstractmethod
    def vec_point(self, t: float, sec: bool=False) -> Vec:
        """Return the position of the track.

        T=0 means the start, T=1 is the end.
        """
        raise NotImplementedError(self)

    @abstractmethod
    def input_norm(self) -> Vec:
        """Return the flow direction at the input point."""

    @abstractmethod
    def output_norm(self) -> Vec:
        """Return the flow direction at the output node."""

    def output_sec_norm(self) -> Vec:
        """Return the flow direction at the secondary output node.

        Only junctions need this to be implemented.
        """
        raise AssertionError(self)

    def tv_name(self) -> str:
        """If a scanner, return the VScript for the scanner prop name."""
        return "null"


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

        model = ent['model'].casefold()
        del ent['model']
        if not model.startswith('models/editor/vactubes/'):
            raise ValueError(
                f'Model "{ent["model"]}" is not a valid vactube '
                f'junction type (at {ent["origin"]})'
            )
        model = model[23:]
        if model == "straight.mdl":
            yield Straight(ent)
        elif model == "curve_1.mdl":
            yield Curve(ent, 64.0, is_reversed)
        elif model == "curve_2.mdl":
            yield Curve(ent, 128.0, is_reversed)
        elif model == "curve_3.mdl":
            yield Curve(ent, 192.0, is_reversed)
        elif model == "curve_4.mdl":
            yield Curve(ent, 256.0, is_reversed)
        elif model == "curve_5.mdl":
            yield Curve(ent, 320.0, is_reversed)
        elif model == "curve_6.mdl":
            yield Curve(ent, 384.0, is_reversed)
        elif model == "splitter_straight.mdl":
            yield Splitter(ent, True)
        elif model == "splitter_sides.mdl":
            yield Splitter(ent, False)
        else:
            raise ValueError(
                f'Model "{ent["model"]}" is not a valid vactube '
                f'junction type (at {ent["origin"]})'
            )


class Spawner(Node):
    """The start point of the track."""
    pass_out_name = 'onspawned'
    keep_ent = True

    def __init__(self, ent: Entity) -> None:
        super().__init__(ent)
        self.speed = srctools.conv_float(ent['speed'], 800.0)
        self.is_auto = srctools.conv_bool(ent['timer'])
        if self.is_auto:
            self.time_min = srctools.conv_float(ent['time_min'], 0.5)
            self.time_max = srctools.conv_float(ent['time_max'], 1.0)
        else:
            self.time_min = self.time_max = 0.0

        # Strip these keyvalues.
        del ent['speed'], ent['timer'], ent['time_min'], ent['time_max']

    def vec_point(self, t: float, sec: bool = False) -> Vec:
        return self.origin

    def path_len(self, sec: bool = False) -> float:
        return 0.0

    def input_norm(self) -> Vec:
        raise AssertionError(self)

    def output_norm(self) -> Vec:
        return Vec(x=1).rotate(*self.angles)


class Destroyer(Node):
    """The end of the track."""
    pass_out_name = 'oncubearrived'

    def vec_point(self, t: float, sec: bool = False) -> Vec:
        return self.origin

    def path_len(self, sec: bool = False) -> float:
        return 0.0

    def input_norm(self) -> Vec:
        return Vec(x=-1).rotate(*self.angles)

    def output_norm(self) -> Vec:
        raise AssertionError(self)


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
        for cube in vmf.by_class['prop_weighted_cube']:
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
        best_cube['allowfunnel'] = '0'
        best_cube['origin'] = ent['origin']

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

    def __init__(self, ent: Entity, radius: float, reversed: bool) -> None:
        super().__init__(ent)
        self.radius = radius
        self.reversed = reversed

    def path_len(self, sec: bool = False) -> float:
        """Return the length of the curve."""
        # πD / 4
        return math.pi * (2.0 / 4.0) * self.radius

    def vec_point(self, t: float, sec: bool = False) -> Vec:
        """Return the position along the curve."""
        assert sec is False

        if self.reversed:
            t = 1.0 - t

        x, y = curve_point(self.radius, t)
        return self.origin + Vec(0, x, -y).rotate(*self.angles)

    def input_norm(self) -> Vec:
        if self.reversed:
            return Vec(z=1).rotate(*self.angles)
        else:
            return Vec(y=1).rotate(*self.angles)

    def output_norm(self) -> Vec:
        if self.reversed:
            return Vec(y=-1).rotate(*self.angles)
        else:
            return Vec(z=-1).rotate(*self.angles)


class Straight(Node):
    """A node pointing directly straight, for outputting or the like.

    This also checks for TV scanners being present nearby.
    """
    SCANNER_TV = ''

    def __init__(self, ent: Entity):
        """Convert the entity to have the right logic."""
        self.scanner = None

        pos = Vec.from_str(ent['origin'])
        for prop in ent.map.by_class['prop_dynamic']:
            if (Vec.from_str(prop['origin']) - pos).mag_sq() > 64**2:
                continue

            model = prop['model'].casefold().replace('\\', '/')
            # Allow spelling this correctly, if you're not Valve.
            if 'vacum_scanner_tv' in model or 'vacuum_scanner_tv' in model:
                self.scanner = prop
                prop.make_unique('_vac_scanner')
                ent.add_out(Output(self.pass_out_name, prop, "Skin", "0", 0.1))
            elif 'vacum_scanner_motion' in model or 'vacuum_scanner_motion' in model:
                prop.make_unique('_vac_scanner')
                ent.add_out(Output(self.pass_out_name, prop, "SetAnimation", "scan01"))

        super(Straight, self).__init__(ent)

    def tv_name(self) -> str:
        """If we have a scanner, return the name of that."""
        if self.scanner is not None:
            return f'"{self.scanner["targetname"]}"'
        else:
            return 'null'

    def path_len(self, sec: bool = False) -> float:
        return 32.0

    def vec_point(self, t: float, sec: bool = False) -> Vec:
        return self.origin + Vec(x=32.0*t - 16.0).rotate(*self.angles)

    def input_norm(self) -> Vec:
        return Vec(x=1.0).rotate(*self.angles)

    def output_norm(self) -> Vec:
        return Vec(x=1.0).rotate(*self.angles)


class Splitter(Node):
    """A T-intersection that either randomly routes cubes or directs them to a dropper."""

    def __init__(self, ent: Entity, straight: bool) -> None:
        """If straight is true, the primary dir goes forward."""
        super().__init__(ent)
        self.is_straight = straight

    def path_len(self, sec: bool = False) -> float:
        if self.is_straight and not sec:
            return 128.0
        else:
            return 64.0 / 4.0 * math.pi

    def vec_point(self, t: float, sec: bool = False) -> Vec:
        x, y = curve_point(64.0, t)
        if sec:
            return self.origin + Vec(y, x, 0).rotate(*self.angles)
        elif self.is_straight:
            return self.origin + Vec(y=128*t).rotate(*self.angles)
        else:
            return self.origin + Vec(-y, x, 0).rotate(*self.angles)

    def input_norm(self) -> Vec:
        return Vec(y=1).rotate(*self.angles)

    def output_norm(self) -> Vec:
        if self.is_straight:
            return Vec(y=1).rotate(*self.angles)
        else:
            return Vec(x=-1).rotate(*self.angles)

    def output_sec_norm(self) -> Vec:
        return Vec(x=1).rotate(*self.angles)
