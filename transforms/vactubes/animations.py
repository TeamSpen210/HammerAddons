"""Handles generating the animation model for the vactubes."""
import math
from typing import Tuple, List, Optional, Union

from . import nodes
from .nodes import DestType
from srctools import Vec, Angle
from srctools.smd import BoneFrame, Mesh
from random import Random


def limit(x: float, num: float) -> float:
    """Clamp x to within ±num."""
    return min(num, max(-num, x))

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

    def __next__(self) -> Tuple[float, float, float]:
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
        # Set of nodes in this animation, to prevent loops.
        self.history: list[nodes.Node] = [start_node]
        # The kind of curve used for the current node.
        self.curve_type = DestType.PRIMARY

        # The source of the cubes on this animation.
        self.start_node = start_node
        # Either the start point, or the splitter to move in the secondary direction.
        self.cur_node: Union[nodes.Spawner, nodes.Splitter] = start_node
        # Once done, the ending node so we can determine if it's a dropper or not.
        self.end_node: Optional[nodes.Destroyer] = None
        # When branching, the amount we overshot into this node from last time.
        self.start_overshoot = 0.0

    def tee(self, split: nodes.Splitter, split_type: DestType, overshoot: float) -> 'Animation':
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
        duplicate.curve_type = split_type
        duplicate.rotator = self.rotator.tee()
        duplicate.move_bone = self.move_bone
        duplicate.cur_frame = self.cur_frame
        duplicate.history = self.history.copy()
        duplicate.pass_points = self.pass_points.copy()

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

    def add_point(self, pos: Vec) -> None:
        """Add the given point to the end of the animation."""
        self.mesh.animation[self.cur_frame] = [
            BoneFrame(self.move_bone, pos, Angle(next(self.rotator)))
        ]
        self.cur_frame += 1


def generate(sources: List[nodes.Spawner]) -> List[Animation]:
    """Generate all the animations, one by one."""
    anims = [Animation(node) for node in sources]

    for anim in anims:
        node = anim.cur_node
        speed = anim.start_node.speed / FPS
        offset = anim.start_node.origin.copy()

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
                if DestType.SECONDARY in node.out_types:
                    anims.append(anim.tee(node, DestType.SECONDARY, overshoot))
                if DestType.TERTIARY in node.out_types:
                    anims.append(anim.tee(node, DestType.TERTIARY, overshoot))

            needs_out = node.has_pass

            seg_len = node.path_len(anim.curve_type)
            seg_frames = math.ceil((seg_len - overshoot) / speed)
            for i in range(int(seg_frames)):
                # Make each frame.
                pos = (overshoot + speed * i) / seg_len
                if needs_out and pos > 0.5:
                    anim.pass_points.append((anim.duration, node))
                    needs_out = False
                # Place the point.
                last_loc = node.vec_point(pos, anim.curve_type)
                anim.add_point(last_loc - offset)

            # If short, we might not have placed the output.
            if needs_out:
                anim.pass_points.append((anim.duration, node))

            # Recalculate the new overshoot.
            overshoot += speed * seg_frames - seg_len

            if isinstance(node, nodes.Destroyer):
                # We reached the end, finalise!
                anim.end_node = node
                anim.add_point(node.origin - offset)
                break

            # Now generate the straight part between this node and the next.
            next_node = node.outputs[anim.curve_type]
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
                    pos = cur_end + ((overshoot + speed * i) / straight_dist) * straight_off
                    anim.add_point(pos - offset)

                overshoot += (speed * seg_frames) - straight_dist
            else:
                overshoot += straight_off.mag()

            # And advance to the next node.
            anim.cur_node = node = next_node
            anim.history.append(node)

            # We only do secondary for the first node, we always continue
            # to the primary value.
            anim.curve_type = DestType.PRIMARY

    return anims
