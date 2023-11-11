"""Entities to allow detecting the motion of vactubes."""
from enum import Enum

import math
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, final

import attrs
from srctools import Vec, conv_float, logger
from srctools.vmf import Entity, Output, VMF

from hammeraddons.bsp_transform.common import RelayOut


LOGGER = logger.get_logger(__name__)


class OutName(Enum):
    """Outputs available on sensors."""
    ENTER = 'onenter'
    MID = 'onpass'
    EXIT = 'onexit'


@final
@attrs.define(eq=False, kw_only=True)
class Sensor:
    """Detects the presence of objects in vactubes."""
    # If from a scanner prop, the relevant entities.
    scanner_tv: Optional[Entity]
    scanner_spinner: Optional[Entity]
    # For each, if their name was initially blank. If we're not used,
    # reset those.
    unnamed_tv: bool = False
    unnamed_spinner: bool = False

    used: bool = False  # Set to indicate this was used.
    radius: float
    origin: Vec

    outputs: Mapping[OutName, Sequence[Output]] = attrs.Factory(dict)
    relays: Dict[OutName, RelayOut] = attrs.field(init=False, factory=dict)

    @classmethod
    def parse(cls, vmf: VMF) -> Iterator['Sensor']:
        """Find all sensor entities in the map."""
        ent: Entity
        for ent in vmf.by_class['comp_vactube_sensor']:
            outputs: Dict[OutName, List[Output]] = {out_kind: [] for out_kind in OutName}
            for out in ent.outputs:
                try:
                    out_kind = OutName(out.output.casefold())
                except ValueError:
                    pass
                else:
                    outputs[out_kind].append(out)

            yield cls(
                scanner_tv=None,
                scanner_spinner=None,

                radius=conv_float(ent['radius'], 16.0),
                origin=Vec.from_str(ent['origin']),
                outputs=outputs,
            )

        spinners: List[Entity] = []
        scanners: List[Sensor] = []

        for ent in vmf.by_class['prop_dynamic']:
            model = ent['model'].casefold().replace('\\', '/')
            # Allow spelling this correctly, if you're not Valve.
            if 'vacum_scanner_tv' in model or 'vacuum_scanner_tv' in model:
                name = ent['targetname']
                was_unnamed = (name == '')
                if was_unnamed:
                    # Give it a unique name so we can target.
                    # If it was named, assume the user took care of making it unique, not up
                    # to us.
                    ent.make_unique('_vac_scanner')
                    name = ent['targetname']
                scanners.append(Sensor(
                    scanner_tv=ent,
                    scanner_spinner=None,
                    unnamed_tv=was_unnamed,

                    radius=48.0,
                    origin=Vec.from_str(ent['origin']),
                    outputs={
                        OutName.ENTER: (),  # Assigning skin is special-cased.
                        OutName.MID: (),
                        OutName.EXIT: (
                            Output('', name, 'Skin', '0'),
                        ),
                    },
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
                    name = spin_ent['targetname']
                    scan.unnamed_spinner = (name == '')
                    if scan.unnamed_spinner:  # Same reasoning as the scanner.
                        spin_ent.make_unique('_vac_spinner')
                        name = spin_ent['targetname']
                    scan.outputs = {
                        **scan.outputs,
                        OutName.ENTER: (
                            *scan.outputs[OutName.ENTER],
                            Output('', name, 'SetAnimation', 'scan01'),
                        )
                    }
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

    def prepare_outputs(self, relay_maker: Iterator[RelayOut]) -> None:
        """If we have been used, generate the relays."""
        if self.used:
            for out_kind, outs in self.outputs.items():
                if not outs:
                    continue
                self.relays[out_kind] = relay = next(relay_maker)
                name = relay.ent['targetname']
                for out in outs:
                    out.output = relay.output
                relay.ent.outputs.extend(outs)
        else:
            if self.scanner_tv is None and self.scanner_spinner is None:
                LOGGER.warning('Vactube sensor at {} did not detect any paths!', self.origin)

            # Not used, reset entity names if required.
            if self.unnamed_tv and self.scanner_tv is not None:
                self.scanner_tv['targetname'] = ''
            if self.unnamed_spinner and self.scanner_spinner is not None:
                self.scanner_spinner['targetname'] = ''
