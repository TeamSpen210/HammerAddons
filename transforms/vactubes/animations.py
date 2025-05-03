"""Handles generating the animation model for the vactubes."""
from collections.abc import Iterator
import math


from . import nodes
from .nodes import Node, DestType
from .sensors import Sensor, OutName as SensorOutput
from srctools import Entity, Vec, Angle, logger
from srctools.smd import BoneFrame, Mesh
from random import Random


def limit(x: float, num: float) -> float:
    """Clamp x to within ±num."""
    return min(num, max(-num, x))

LOGGER = logger.get_logger(__name__)
SKIN_INPUT = '<SKIN>'  # Special input name which sets skin appropriately.
# Max angular acceleration per frame
ROT_ACC = 4.0
# Max angular speed per frame
ROT_LIM = 30.0

FPS = 30


class RotGen:
    """Generate a stream of random rotations."""
    def __init__(self, seed: str, pitch: float=0.0, yaw: float=0.0, roll: float=0.0) -> None:
        self.rand = Random()
        # If a valid hex number, use that way. Otherwise use the bytes.
        try:
            seed_num = int(seed, 16)
        except ValueError:
            self.rand.seed(seed.encode('ascii', 'replace'))
        else:
            self.rand.seed(seed_num)

        self.pit = self.yaw = self.rol = 0.0
        self.pit_sp = pitch
        self.yaw_sp = yaw
        self.rol_sp = roll

    def __next__(self) -> tuple[float, float, float]:
        self.pit_sp = limit(
            self.pit_sp + self.rand.uniform(-ROT_ACC, ROT_ACC), ROT_LIM)

        self.yaw_sp = limit(
            self.yaw_sp + self.rand.uniform(-ROT_ACC, ROT_ACC), ROT_LIM)

        self.rol_sp = limit(
            self.rol_sp + self.rand.uniform(-ROT_ACC, ROT_ACC), ROT_LIM)

        self.pit = (self.pit + self.pit_sp) % 360
        self.yaw = (self.yaw + self.yaw_sp) % 360
        self.rol = (self.rol + self.rol_sp) % 360

        return self.pit, self.yaw, self.rol

    def tee(self: 'RotGen') -> 'RotGen':
        """Duplicate this rotator, so the clone maintains continuity with the last frame output."""
        # Use our random to produce the seed. This means the clone will
        # produce distinct values compared to us, but will still ultimately
        # be determined by the original seed.
        return RotGen(
            format(self.rand.getrandbits(64), 'X'),
            self.pit_sp, self.yaw_sp, self.rol_sp,
        )


class Animation:
    """Represents a single animated path being generated."""
    def __init__(self, start_node: nodes.Spawner) -> None:
        self.mesh = Mesh.blank('root')
        self.name = ''  # Set later in main transform logic.
        self.rotator = RotGen(start_node.seed)
        [self.move_bone] = self.mesh.bones.values()
        self.cur_frame = 0
        # For nodes with OnPass outputs, the time to fire each of those.
        self.pass_points: list[tuple[float, nodes.Node]] = []
        # For sensors that we're currently inside, the time at which we entered them.
        self.sensor_enter: dict[Sensor, float] = {}
        # Sensors we have passed through, and the start/end time.
        self.sensors: list[tuple[float, float, Sensor]] = []
        # Set of nodes in this animation, to prevent loops.
        self.history: list[nodes.Node] = [start_node]
        # The kind of curve used for the current node.
        self.curve_type = DestType.PRIMARY

        # The animation starts at 0 0 0 in the file.
        self.start_pos = start_node.origin.freeze()

        # The source of the cubes on this animation.
        self.start_node = start_node
        # Either the start point, or the splitter to move in the secondary direction.
        self.cur_node: nodes.Node = start_node
        # Once done, this is the ending node so that we can determine if it's a dropper or not.
        self.end_node: nodes.Destroyer | None = None
        # When branching, the amount we overshot into this node from last time.
        self.start_overshoot = 0.0

    def tee(self, split: nodes.Node, split_type: DestType, overshoot: float) -> 'Animation':
        """Duplicate this animation so additional frames can be added.

        Note: Does not fully copy, the existing frame data is shared so
        don't modify previous locations!
        """
        duplicate = self.__new__(Animation)
        duplicate.mesh = Mesh(
            self.mesh.bones,
            self.mesh.animation.copy(),
            [],
        )
        duplicate.start_pos = self.start_pos
        duplicate.curve_type = split_type
        duplicate.rotator = self.rotator.tee()
        duplicate.move_bone = self.move_bone
        duplicate.cur_frame = self.cur_frame
        duplicate.history = self.history.copy()
        duplicate.pass_points = self.pass_points.copy()
        duplicate.sensor_enter = self.sensor_enter.copy()
        duplicate.sensors = self.sensors.copy()

        duplicate.start_node = self.start_node
        duplicate.cur_node = split
        duplicate.end_node = None
        assert self.end_node is None
        duplicate.start_overshoot = overshoot
        return duplicate

    @property
    def duration(self) -> float:
        """Return the current duration of the animation."""
        return self.cur_frame / FPS

    def add_point(self, sensors: list[Sensor], pos: Vec) -> None:
        """Add the given point to the end of the animation."""
        if self.cur_frame != 0:  # Handle sensors, if we're not the first frame.
            previous = self.mesh.animation[self.cur_frame - 1][0].position
            self.check_sensors(sensors, previous + self.start_pos, pos)
        self.mesh.animation[self.cur_frame] = [
            BoneFrame(self.move_bone, pos - self.start_pos, Angle(next(self.rotator)))
        ]
        self.cur_frame += 1

    def check_sensors(self, sensors: list[Sensor], pos1: Vec, pos2: Vec) -> None:
        """Check all our sensors, and update values depending on them."""
        if not sensors:  # No sensors, nothing to do.
            return
        direction = pos2 - pos1
        dist = direction.mag()
        direction /= dist
        for sensor in sensors:
            intersect = sensor.intersect(pos1, direction, dist)
            if intersect is not None:
                # We hit the sensor. If we were already inside, check if we're leaving.
                a, b = intersect
                if sensor in self.sensor_enter:
                    start_time = self.sensor_enter[sensor]
                    if 0 <= b <= dist:
                        # We're passing out of the sensor.
                        end_time = (self.cur_frame - 1.0 + b / dist) / FPS
                        self.sensors.append((start_time, end_time, sensor))
                        sensor.used = True
                        del self.sensor_enter[sensor]
                    # Else, we are still inside, so nothing to do.
                elif 0 <= a <= dist:
                    # Just entered the sensor.
                    start_time = (self.cur_frame + a / dist) / FPS
                    if 0 <= b <= dist:
                        # Special case - entered and exited the same frame.
                        end_time = (self.cur_frame + b / dist) / FPS
                        self.sensors.append((start_time, end_time, sensor))
                        sensor.used = True
                    else:
                        self.sensor_enter[sensor] = start_time
            elif sensor in self.sensor_enter:
                # Not intersecting, but was last time. We must have left, assume at the start of
                # last frame.
                start_time = self.sensor_enter.pop(sensor)
                end_time = (self.cur_frame - 1) / FPS
                self.sensors.append((start_time, end_time, sensor))
                sensor.used = True

    def vscript_outputs(self) -> Iterator[tuple[float, Entity, str]]:
        """Generate the names the VScript should call."""
        for time, node in self.pass_points:
            if node.pass_relay is not None:
                yield time, node.pass_relay.ent, node.pass_relay.input
        for enter, exit, sensor in self.sensors:
            try:
                relay = sensor.relays[SensorOutput.ENTER]
            except KeyError:
                pass
            else:
                yield enter, relay.ent, relay.input
            try:
                relay = sensor.relays[SensorOutput.EXIT]
            except KeyError:
                pass
            else:
                yield exit, relay.ent, relay.input
            try:
                relay = sensor.relays[SensorOutput.MID]
            except KeyError:
                pass
            else:
                yield (enter + exit) / 2, relay.ent, relay.input

            if sensor.used and sensor.scanner_tv is not None:
                yield enter, sensor.scanner_tv, SKIN_INPUT


def generate(sources: list[nodes.Spawner], sensors: list[Sensor]) -> list[Animation]:
    """Generate all the animations, one by one."""
    anims = [Animation(spawner) for spawner in sources]

    for anim in anims:
        node: Node = anim.cur_node
        speed = anim.start_node.speed / FPS

        # To keep the speed constant, keep track of any extra we need to offset
        # into the next node.
        overshoot = anim.start_overshoot

        # To generate, we alternate between making a single node, and then
        # making the straight section.

        while True:
            # First, check to see if we need to branch off.
            # If we're secondary, we are the branch off and so don't need to
            # do that again.
            if anim.curve_type is DestType.PRIMARY:
                # Intentionally adding these, we'll iterate over them later.
                if DestType.SECONDARY in node.out_types:
                    anims.append(anim.tee(node, DestType.SECONDARY, overshoot))  # noqa: B909
                if DestType.TERTIARY in node.out_types:
                    anims.append(anim.tee(node, DestType.TERTIARY, overshoot))  # noqa: B909

            needs_out = node.pass_relay is not None

            seg_len = node.path_len(anim.curve_type)
            seg_frames = math.ceil((seg_len - overshoot) / speed)
            for i in range(int(seg_frames)):
                # Make each frame.
                fraction = (overshoot + speed * i) / seg_len
                if needs_out and fraction > 0.5:
                    anim.pass_points.append((anim.duration, node))
                    needs_out = False
                # Place the point.
                last_loc = node.vec_point(fraction, anim.curve_type)
                anim.add_point(sensors, last_loc)

            # If short, we might not have placed the output.
            if needs_out:
                anim.pass_points.append((anim.duration, node))

            # Recalculate the new overshoot.
            overshoot += speed * seg_frames - seg_len

            if isinstance(node, nodes.Destroyer):
                # We reached the end, finalise!
                anim.end_node = node
                anim.add_point(sensors, node.origin)
                break

            # Now generate the straight part between this node and the next.
            next_node = node.outputs[anim.curve_type]
            assert next_node is not None
            cur_end = node.vec_point(1.0, anim.curve_type)
            straight_off = next_node.vec_point(0.0) - cur_end

            if next_node in anim.history:
                raise ValueError(
                    f'Vactube junction "{next_node.name}" at {next_node.origin} '
                    f"loops back onto itself!"
                )

            # Only generate the straight part if the nodes aren't overlapping.
            if Vec.dot(straight_off, next_node.input_norm()) > 0:
                straight_dist = straight_off.mag()
                seg_frames = math.ceil((straight_dist - overshoot) / speed)

                for i in range(int(seg_frames)):
                    # Make each frame.
                    pos = cur_end + straight_off * ((overshoot + speed * i) / straight_dist)
                    anim.add_point(sensors, pos)

                overshoot += (speed * seg_frames) - straight_dist
            else:
                overshoot += straight_off.mag()

            # And advance to the next node.
            anim.cur_node = next_node
            node = next_node
            anim.history.append(node)

            # We only do secondary for the first node, we always continue
            # to the primary value.
            anim.curve_type = DestType.PRIMARY

    return anims
