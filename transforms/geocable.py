""""Compile static prop cables, instead of sprites."""
import math
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    Optional, List, Tuple, NamedTuple, FrozenSet,
    TypeVar, MutableMapping, NewType, Set, Iterable, Dict, Iterator,
)

from srctools import Vec, Entity, conv_int, conv_float, logger, Matrix
from srctools.compiler.mdl_compiler import ModelCompiler
from srctools.bsp_transform import Context, trans
from srctools.bsp import StaticProp, StaticPropFlags
from srctools.smd import Mesh, Vertex, Triangle


LOGGER = logger.get_logger(__name__)
NodeID = NewType('NodeID', str)
Number = TypeVar('Number', int, float)

QC_TEMPLATE = '''\
$staticprop
$modelname "{path}"

$body body "cable.smd"
$cdmaterials ""
$sequence idle "cable.smd" act_idle 1
'''


def lerp(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linearly interpolate from in to out."""
    return out_min + (((x - in_min) * (out_max - out_min)) / (in_max - in_min))


class InterpType(Enum):
    """Type of interpolation to use."""
    STRAIGHT = 0
    CATMULL_ROM = 1
    ROPE = 2


class Config(NamedTuple):
    """Configuration specified in rope entities. This can be shared to reduce duplication."""
    material: str
    segments: int
    side_count: int
    radius: float
    interp: InterpType
    slack: float

    @staticmethod
    def _parse_min(ent: Entity, keyvalue: str, minimum: Number, message: str) -> Number:
        """Helper for passing all the numeric keys."""
        value = (conv_float if isinstance(minimum, float) else conv_int)(ent[keyvalue], minimum)
        if value < minimum:
            LOGGER.warning(message, ent['origin'])
            return minimum
        return value

    @classmethod
    def parse(cls, ent: Entity) -> 'Config':
        """Parse from an entity."""
        segments = cls._parse_min(
            ent, 'segments', 0,
            'Segment count for rope at '
            '{} must be positive or zero!'
        )
        side_count = cls._parse_min(
            ent, 'sides', 3,
            'Ropes cannot have less than 3 sides! (node at {})',
        )
        radius = cls._parse_min(
            ent, 'radius', 0.1,
            'Radius for rope at {} must be positive!',
        )
        slack = cls._parse_min(
            ent, 'slack', 0.0,
            'Rope at {} cannot have a negative slack!',
        )
        try:
            interp_type = InterpType(int(ent['interpolationtype', '2']))
        except ValueError:
            LOGGER.warning(
                'Unknown interpolation type "{}" '
                'for rope at {}!',
                ent['interpolationtype'],
                ent['origin'],
            )
            interp_type = InterpType.STRAIGHT

        return cls(
            ent['material'],
            segments,
            side_count,
            radius,
            interp_type,
            slack,
        )


class NodeEnt:
    """A node entity, and its associated configuration. This is used to match with earlier compiles."""
    def __init__(
        self,
        pos: Vec,
        config: Config,
        node_id: NodeID,
        group: str,
    ) -> None:
        self.id = node_id
        self.config = config
        self.group = group  # Nodes with the same group compile together.
        self.pos = pos

    def __repr__(self) -> str:
        return f'<NodeEnt "{self.id}" @ {self.pos}>'

    def __hash__(self) -> int:
        """Allow this to be hashed."""
        return hash((
            self.id,
            self.pos.x, self.pos.y, self.pos.z,
            self.config,
        ))

    def __eq__(self, other: object) -> object:
        """Allow this to be compared."""
        if isinstance(other, NodeEnt):
            return (
                self.id == other.id and
                self.pos == other.pos and
                self.config == other.config
            )
        return NotImplemented

    def __ne__(self, other: object) -> object:
        """Allow this to be compared."""
        if isinstance(other, NodeEnt):
            return (
                self.id != other.id or
                self.pos != other.pos or
                self.config != other.config
            )
        return NotImplemented


class Node:
    """All the data for a node, used during constrction of the geo.

    Unlike BasicNode, this is compared by identity, and has no ID.
    """
    def __init__(self, pos: Vec, config: Config) -> None:
        self.config = config
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None
        self.pos = pos
        # Orientation of the segment up to the next.
        self.orient = Matrix()
        # The points for the cylinder, on these sides.
        self.points_prev: List[Vertex] = []
        self.points_next: List[Vertex] = []

    @classmethod
    def from_ent(cls, ent: NodeEnt) -> 'Node':
        return Node(ent.pos.copy(), ent.config)

    def clone(self) -> 'Node':
        """Create a duplicate of this node, but with no connections."""
        return Node(self.pos.copy(), self.config)

    def __repr__(self) -> str:
        return f'<Node at {self.pos}>'


def build_rope(
    nodes_and_conn: Tuple[FrozenSet[NodeEnt], FrozenSet[Tuple[NodeID, NodeID]]],
    temp_folder: Path,
    mdl_name: str,
) -> None:
    """Construct the geometry for a rope."""
    LOGGER.info('Building rope {}', mdl_name)
    ents, connections = nodes_and_conn
    mesh = Mesh.blank('root')
    [bone] = mesh.bones.values()
    bone_weight = [(bone, 1.0)]

    id_to_node = {
        node.id: Node(node.pos.copy(), node.config)
        for node in ents
    }
    nodes: Set[Node] = set(id_to_node.values())

    # First connect all the nodes.
    for id1, id2 in connections:
        first = id_to_node[id1]
        second = id_to_node[id2]
        assert first.next is None, (first, second)
        assert second.prev is None, (first, second)
        first.next = second
        second.prev = first

    interpolate_all(nodes)
    for node in nodes:
        if node.next is None:
            continue
        forward = node.orient.forward()
        u = node.orient.left()
        v = node.orient.up()

        count = node.config.side_count
        mat = node.config.material
        delta = 2 * math.pi / count
        radius_a = node.config.radius
        radius_b = node.next.config.radius
        for i in range(count):
            left_ang = delta * i
            right_ang = delta * (i + 1)
            left = u * math.cos(left_ang) + v * math.sin(left_ang)
            right = u * math.cos(right_ang) + v * math.sin(right_ang)
            left_a = Vertex(node.pos + radius_a * left, left, 0.0, 0.0, bone_weight)
            right_a = Vertex(node.pos + radius_a * right, right, 1.0, 0.0, bone_weight)
            left_b = Vertex(node.next.pos + radius_b * left, left, 0.0, 1.0, bone_weight)
            right_b = Vertex(node.next.pos + radius_b * right, right, 1.0, 1.0, bone_weight)
            mesh.triangles.append(Triangle(mat, left_a, right_b, left_b))
            mesh.triangles.append(Triangle(mat, left_a, right_a, right_b))

    with (temp_folder / 'cable.smd').open('wb') as fb:
        mesh.export(fb)

    with (temp_folder / 'model.qc').open('w') as f:
        f.write(QC_TEMPLATE.format(path=mdl_name))

def interpolate_straight(node1: Node, node2: Node, seg_count: int) -> List[Node]:
    """Simply interpolate in a straight line."""
    diff = (node2.pos - node1.pos) / (seg_count + 1)
    return [
        Node(node1.pos + diff * i, node1.config)
        for i in range(1, seg_count + 1)
    ]


def interpolate_catmull_rom(node1: Node, node2: Node, seg_count: int) -> List[Node]:
    """Interpolate a spline curve, matching Valve's implementation."""
    # If no points are found, extrapolate out the line.
    diff = (node2.pos - node1.pos).norm()
    if node1.prev is None:
        p0 = node1.pos - diff
    else:
        p0 = node1.prev.pos
    p1 = node1.pos
    p2 = node2.pos
    if node2.next is None:
        p3 = node2.pos + diff
    else:
        p3 = node2.next.pos
    t0 = 0
    t1 = t0 + (p1-p0).mag()
    t2 = t1 + (p2-p1).mag()
    t3 = t2 + (p3-p2).mag()
    points: List[Node] = []
    for i in range(1, seg_count):
        t = lerp(i, 0, seg_count, t1, t2)
        A1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
        A2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2
        A3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3

        B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
        B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

        points.append(Node(
            (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2,
            node1.config,
        ))
    return points


def interpolate_rope(node1: Node, node2: Node, seg_count: int) -> List[Node]:
    """Compute the move_rope style hanging points."""
    # TODO: Rope curve
    return interpolate_catmull_rom(node1, node2, seg_count)


def interpolate_all(nodes: Set[Node]) -> None:
    """Produce nodes in-between each user-made node."""
    # Create the nodes and put them in a seperate list, then add them
    # to the actual nodes list second. This way sections that have been interpolated
    # don't affect the interpolation of neighbouring sections.

    segments: List[List[Node]] = []
    for node1 in nodes:
        if node1.next is None or node1.config.segments <= 0:
            continue
        node2 = node1.next
        interp_type = node1.config.interp
        func = globals()['interpolate_' + interp_type.name.casefold()]
        points = func(node1, node2, node1.config.segments)

        for a, b in zip(points, points[1:]):
            a.next = b
            b.prev = a
        points[0].prev = node1
        points[-1].next = node2
        segments.append(points)

    for points in segments:
        nodes.update(points)
        points[0].prev.next = points[0]
        points[-1].next.prev = points[-1]
@trans('Model Ropes')
def comp_prop_rope(ctx: Context) -> None:
    """Build static props for ropes."""
    compiler = ModelCompiler.from_ctx(ctx, 'ropes')
    # group -> id -> node.
    all_nodes: MutableMapping[str, MutableMapping[NodeID, NodeEnt]] = defaultdict(dict)
    # Given a targetname, all the nodes with that name.
    name_to_ids: MutableMapping[str, List[NodeEnt]] = defaultdict(list)
    # Store the node/next-key pairs for linking after they're all parsed.
    temp_conns: List[Tuple[NodeEnt, str]] = []

    for ent in ctx.vmf.by_class['comp_prop_rope'] | ctx.vmf.by_class['comp_prop_cable']:
        ent.remove()
        conf = Config.parse(ent)
        node = NodeEnt(
            Vec.from_str(ent['origin']),
            conf,
            NodeID(ent['hammerid']),
            ent['group'].casefold(),
        )
        all_nodes[node.group][node.id] = node
        ctx.vmf.create_ent(
            'env_sprite',
            origin=node.pos,
            model='sprites/glow01.spr',
            spawnflags='1',
            scale=0.25,
            glowproxysize=2,
        )

        if ent['targetname']:
            name_to_ids[ent['targetname'].casefold()].append(node)
        if ent['nextkey']:
            temp_conns.append((node, ent['nextkey'].casefold()))

    if not all_nodes:
        return
    LOGGER.info('{} rope nodes found.', len(all_nodes))

    connections: MutableMapping[str, Set[Tuple[NodeID, NodeID]]] = defaultdict(set)

    for node, target in temp_conns:
        found = []
        if target.endswith('*'):
            search = target[:-1]
            for name, nodes in name_to_ids.items():
                if name.startswith(search):
                    found.extend(nodes)
        else:
            found.extend(name_to_ids.get(target, ()))
        found.sort()
        for dest in found:
            if node.group != dest.group:
                raise ValueError(
                    'Ropes have differing groups: {} @ {}, {} @ {}',
                    node.group, node.pos,
                    dest.group, dest.pos,
                )
            connections[node.group].add((node.id, dest.id))

    # TODO, compute the visleafs.
    static_props = list(ctx.bsp.static_props())
    all_leafs = set()
    for prop in static_props:
        all_leafs.update(prop.visleafs)

    with compiler:
        for group, nodes in all_nodes.items():
            bbox_min, bbox_max = Vec.bbox(node.pos for node in nodes.values())
            center = (bbox_min + bbox_max) / 2
            for node in nodes.values():
                node.pos -= center

            model_name = compiler.get_model(
                (frozenset(nodes.values()), frozenset(connections[group])),
                build_rope,
            )
            static_props.append(StaticProp(
                model=model_name,
                origin=center,
                angles=Vec(0, 270, 0),
                scaling=1.0,
                visleafs=list(all_leafs),
                solidity=0,
            ))
    LOGGER.info('Built {} models.', len(all_nodes))
    ctx.bsp.write_static_props(static_props)
