"""Entities to allow detecting the motion of vactubes."""
import math
from typing import Iterator, List, Optional, Sequence, Tuple, final

import attrs

from srctools import Vec, conv_float
from srctools.vmf import Entity, Output, VMF

SPIN_ANIM = "scan01"


@final
@attrs.define(eq=False, kw_only=True)
class Sensor:
    """Detects the presence of objects in vactubes."""
    # If from a scanner prop, the relevant entities.
    scanner_tv: Optional[Entity]
    scanner_spinner: Optional[Entity]

    used: bool = False  # Set to indicate this was detected.
    radius: float
    origin: Vec

    out_on_enter: Sequence[Output]
    out_on_exit: Sequence[Output]
    out_on_mid: Sequence[Output]

    @classmethod
    def parse(cls, vmf: VMF) -> Iterator['Sensor']:
        """Find all sensor entities in the map."""
        ent: Entity
        for ent in vmf.by_class['comp_vactube_sensor']:
            out_enter = []
            out_exit = []
            out_mid = []
            for out in ent.outputs:
                name = out.output.casefold()
                if name == 'onenter':
                    out_enter.append(out)
                elif name == 'onpass':
                    out_mid.append(out)
                elif name == 'onexit':
                    out_exit.append(out)

            yield cls(
                scanner_tv=None,
                scanner_spinner=None,

                radius=conv_float(ent['radius'], 16.0),
                origin=Vec.from_str(ent['origin']),
                out_on_enter=out_enter,
                out_on_mid=out_mid,
                out_on_exit=out_exit,
            )

        spinners: List[Entity] = []
        scanners: List[Sensor] = []

        for ent in vmf.by_class['prop_dynamic']:
            model = ent['model'].casefold().replace('\\', '/')
            # Allow spelling this correctly, if you're not Valve.
            if 'vacum_scanner_tv' in model or 'vacuum_scanner_tv' in model:
                scanners.append(Sensor(
                    scanner_tv=ent,
                    scanner_spinner=None,

                    radius=48.0,
                    origin=Vec.from_str(ent['origin']),
                    out_on_enter=(),
                    out_on_mid=(),
                    out_on_exit=(),
                ))
            elif 'vacum_scanner_motion' in model or 'vacuum_scanner_motion' in model:
                spinners.append(ent)

        for spin_ent in spinners:
            pos = Vec.from_str(spin_ent['origin'])
            # To match, they need to be within 8 units (arbitrarily).
            for scan in scanners:
                if scan.scanner_spinner is None and (pos - scan.origin).mag_sq() <= 40 ** 2:
                    # Found.
                    scan.scanner_spinner = spin_ent
                    break
            # else: isolated, not important.
        # Only yield now they're linked.
        yield from scanners

    def intersect(self, start: Vec, direction: Vec, dist: float) -> Optional[Tuple[float, float]]:
        """Check if a line segment intersects this sensor.

        The line is defined by `start + direction * dist`. The return value is None if it doesn't
        intersect, or a (start, end) fraction where both are between [0, dist] and start < end. If the
        line segment actually touches at a single point, it's expanded into a 1 unit line.
        """
        off = start - self.origin
        dot = Vec.dot(direction, off)
        delta = dot**2 - (off.mag_sq() - self.radius**2)
        if delta < -1e-9:
            return None
        elif delta > 1e-9:
            # Two solutions.
            root = math.sqrt(delta)
            pos1 = -dot-root
            pos2 = -dot+root
            if pos2 < 0.0 or pos1 > dist:
                # The infinite line intersects, but not the part we care about.
                return None
            return pos1, pos2
        else:
            # One solution, treat as two.
            return -dot - 0.5, -dot + 0.5
