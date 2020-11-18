""""Compile static prop cables, instead of sprites."""
import math
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    Optional, List, Tuple, NamedTuple, FrozenSet,
    TypeVar, MutableMapping, NewType, Set, Iterable, Dict, Iterator,
)

from srctools import (
    logger, conv_int, conv_float, conv_bool,
    Vec, Entity, Matrix,
)
from srctools.compiler.mdl_compiler import ModelCompiler
from srctools.bsp_transform import Context, trans
from srctools.bsp import StaticProp, StaticPropFlags
from srctools.smd import Mesh, Vertex, Triangle, Bone


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

class RopePhys:
    """Holds the data for move_rope simulation."""
    __slots__ = ['pos', 'prev_pos']
    pos: Vec
    prev_pos: Vec

    def __init__(self, pos: Vec) -> None:
        self.pos = pos
        self.prev_pos = pos.copy()


ROPE_GRAVITY = -1500
SIM_TIME = 5.00
TIME_STEP = 1/50


class Config(NamedTuple):
    """Configuration specified in rope entities. This can be shared to reduce duplication."""
    material: str
    segments: int
    side_count: int
    radius: float
    interp: InterpType
    slack: float
    u_min: float
    u_max: float
    v_scale: float
    flip_uv: bool

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

        v_scale = abs(conv_float(ent['mat_scale'], 1.0))
        u_min = abs(conv_float(ent['u_min'], 0.0))
        u_max = abs(conv_float(ent['u_max'], 1.0))
        # Rescale this, so that if it's 1, the pixels are square.
        v_scale *= (u_max - u_min) / (2*math.pi*radius)

        return cls(
            ent['material'],
            segments,
            side_count,
            radius,
            interp_type,
            slack,
            u_min, u_max,
            v_scale,
            conv_bool(ent['mat_rotate']),
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

    def find_start(self) -> 'Node':
        """Find the start of this chain, or return self if it's a loop."""
        node = self
        while node.prev is not None and node.prev is not self:
            node = node.prev
        return node

    def follow(self) -> Iterator['Node']:
        """Iterate over every node."""
        yield self
        node = self.next
        while node is not None and node is not self:
            yield node
            node = node.next

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

    nodes = build_node_tree(ents, connections)

    interpolate_all(nodes)
    compute_orients(nodes)
    compute_verts(nodes, bone)

    generate_straights(nodes, mesh)
    generate_caps(nodes, mesh)

    # Move the UVs around so they don't extend too far.
    for tri in mesh.triangles:
        u = math.floor(min(point.tex_u for point in tri))
        v = math.floor(min(point.tex_v for point in tri))
        if u or v:
            tri.point1 = tri.point1.with_uv(tri.point1.tex_u - u, tri.point1.tex_v - v)
            tri.point2 = tri.point2.with_uv(tri.point2.tex_u - u, tri.point2.tex_v - v)
            tri.point3 = tri.point2.with_uv(tri.point3.tex_u - u, tri.point3.tex_v - v)

    with (temp_folder / 'cable.smd').open('wb') as fb:
        mesh.export(fb)

    with (temp_folder / 'model.qc').open('w') as f:
        f.write(QC_TEMPLATE.format(path=mdl_name))


def build_node_tree(ents: FrozenSet[NodeEnt], connections: FrozenSet[Tuple[NodeID, NodeID]]) -> Set[Node]:
    """Convert the ents/connections definitions into a node tree."""
    # Convert them all into the real node objects.
    id_to_node = {
        node.id: Node(node.pos.copy(), node.config)
        for node in ents
    }
    nodes: Set[Node] = set(id_to_node.values())

    def maybe_split(node: Node, attr: str) -> Node:
        """Split nodes to ensure they only have 1 or 2 connections.

        If it has more, or multiple in one side, it will be converted
        to multiple theat end at the same point.
        """
        if node not in nodes:  # Already split, create a copy and return.
            copy = node.clone()
            nodes.add(copy)
            return copy
        if getattr(node, attr) is not None:
            # Need to split this one.
            if node.next is not None:
                forward = node.clone()
                nodes.add(forward)
                forward.next = node.next
                node.next.prev = forward
            if node.prev is not None:
                reverse = node.clone()
                nodes.add(reverse)
                reverse.prev = node.prev
                node.prev.next = reverse
            node.prev = node.next = None
            copy = node.clone()
            nodes.add(copy)
            return copy
        return node

    for id1, id2 in connections:
        first = maybe_split(id_to_node[id1], "next")
        second = maybe_split(id_to_node[id2], "prev")

        first.next = second
        second.prev = first

    return nodes


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
    """Compute the move_rope style hanging points.

    This uses a quite unusual implementation in Source, doing a physics simulation.
    See the following files for this code:
        - src/public/keyframe.cpp
        - src/public/simple_physics.cpp
        - src/public/rope_physics.cpp
    """
    diff = node2.pos - node1.pos
    total_len = diff.mag() + max(0.0, node1.config.slack - 100.0)
    max_len = total_len / (seg_count + 1)
    max_len_sqr = max_len ** 2

    interp_diff = diff / (seg_count + 1)
    points = [
        RopePhys(node1.pos + interp_diff * i)
        for i in range(0, seg_count + 2)
    ]
    springs = list(zip(points, points[1:]))

    time = 0
    step = TIME_STEP
    gravity = Vec(z=ROPE_GRAVITY) * step**2

    # Start/end doesn't move.
    anchor1, *moveable, anchor2 = points

    while time < SIM_TIME:
        time += step
        points[0].pos = node1.pos.copy()
        points[-1].pos = node2.pos.copy()
        # Gravity.
        for node in moveable:
            node.prev_pos, node.pos = node.pos, (
                node.pos + (node.pos - node.prev_pos) * 0.98 + gravity
            )

        # Spring constraints.
        for _ in range(3):
            for phys1, phys2 in springs:
                diff = phys1.pos - phys2.pos
                dist = diff.mag_sq()
                if dist > max_len_sqr:
                    diff *= 0.5 * (1 - (max_len / math.sqrt(dist)))
                    if phys1 is not anchor1:
                        phys1.pos -= diff
                    if phys2 is not anchor2:
                        phys2.pos += diff

    return [
        Node(point.pos, node1.config)
        for point in moveable
    ]


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

    # Finally, split nodes with too much of an angle between them - we can't smooth.
    for node in list(nodes):
        if node.prev is None or node.next is None:
            continue
        off1 = node.pos - node.prev.pos
        off2 = node.next.pos - node.pos
        if Vec.dot(off1, off2) < 0.7:
            new_node = Node(node.pos.copy(), node.config)
            nodes.add(new_node)
            new_node.next = node.next
            node.next.prev = new_node
            node.next = None


def compute_orients(nodes: Iterable[Node]) -> None:
    """Compute the appropriate orientation for each node."""
    # This is based on the info at:
    # https://janakiev.com/blog/framing-parametric-curves/
    tangents: Dict[Node, Vec] = {}
    all_nodes: Set[Node] = set()
    for node in nodes:
        if node.prev is node.next is None:
            continue
        node_prev = node.prev if node.prev is not None else node
        node_next = node.next if node.next is not None else node
        tangents[node] = (node_next.pos - node_prev.pos).norm()
        all_nodes.add(node)

    while all_nodes:
        node1 = all_nodes.pop()
        node1 = node1.find_start()
        tanj1 = tangents[node1]
        up = Vec(tanj1.y, -tanj1.z, 0).norm()
        if not up:  # Only occurs if pointing vertical.
            up = Vec(1, 0, 0)
        node1.orient = Matrix.from_basis(x=tanj1, z=up)
        while node1.next is not None:
            node2 = node1.next
            all_nodes.discard(node2)
            tanj1 = tangents[node1]
            tanj2 = tangents[node2]
            b = Vec.cross(tanj1, tanj2)
            if b.mag_sq() < 0.001:
                node2.orient = node1.orient.copy()
            else:
                b = b.norm()
                phi = math.acos(Vec.dot(tanj1, tanj2))
                up = node1.orient.up() @ Matrix.axis_angle(b, -math.degrees(phi))
                node2.orient = Matrix.from_basis(x=tanj2, z=up)
            node1 = node2


def compute_verts(nodes: Iterable[Node], bone: Bone) -> None:
    """Build the initial vertexes of each node."""
    bone_weight = [(bone, 1.0)]
    todo = set(nodes)

    def vert(pos: Vec, off: Vec, u: float, v: float) -> Vertex:
        """Make a vertex, with appropriate UVs."""
        if config.flip_uv:
            v, u = u, v
        return Vertex(pos + off, off.norm(), u, v, bone_weight)

    while todo:
        start = todo.pop()
        start = start.find_start()
        if start.next is None:
            continue
        v_start = 0.0
        for node1 in start.follow():
            todo.discard(node1)
            node2 = node1.next if node1.next is not None else node1
            config = node1.config
            count = node1.config.side_count
            radius_a = node1.config.radius
            radius_b = node2.config.radius
            v_end = v_start + config.v_scale * (node2.pos - node1.pos).mag()
            for i in range(count):
                ang = lerp(i, 0, count, 0, 2*math.pi)
                local = Vec(0, math.cos(ang), math.sin(ang))
                u = lerp(i, 0, count, config.u_min, config.u_max)
                node1.points_next.append(vert(node1.pos, radius_a * local @ node1.orient, u, v_start))
                if node1.next is not None:
                    node1.next.points_prev.append(vert(node2.pos, radius_b * local @ node2.orient, u, v_end))
            v_start = v_end


def generate_straights(nodes: Iterable[Node], mesh: Mesh) -> None:
    """Finally, generate all the straight-side sections."""
    for node1 in nodes:
        node2 = node1.next
        if node2 is None:
            continue
        side_count = node1.config.side_count
        mat = node1.config.material
        for i in range(node1.config.side_count):
            left_a = node1.points_next[i]
            right_a = node1.points_next[(i + 1) % side_count]
            left_b = node2.points_prev[i % side_count].copy()
            right_b = node2.points_prev[(i + 1) % side_count]
            
            # If it flips around, we need to fix that.
            if right_a is node1.points_next[0]:
                right_a = right_a.copy()
                if node1.config.flip_uv:
                    right_a.tex_v = node1.config.u_max
                else:
                    right_a.tex_u = node1.config.u_max
            if right_b is node2.points_prev[0]:
                right_b = right_b.copy()
                if node2.config.flip_uv:
                    right_b.tex_v = node2.config.u_max
                else:
                    right_b.tex_u = node2.config.u_max
            
            mesh.triangles.append(Triangle(mat, left_a, right_b, left_b))
            mesh.triangles.append(Triangle(mat, left_a, right_a, right_b))


def generate_caps(nodes: Iterable[Node], mesh: Mesh) -> None:
    """Cap off any unfinished sides.

    We just use a simple fan layout.
    """
    def make_cap(orig, norm):
        # Recompute the UVs to use the first bit of the cable.
        points = [
            Vertex(
                point.pos, norm,
                lerp(Vec.dot(point.norm, node.orient.up()), -1, 1, node.config.u_min, node.config.u_max),
                lerp(Vec.dot(point.norm, node.orient.left()), -1, 1, 0, v_max),
                point.links,
            )
            for point in orig
        ]
        mesh.triangles.append(Triangle(mat, points[0], points[1], points[2]))
        for a, b in zip(points[2:], points[3:]):
            mesh.triangles.append(Triangle(mat, points[0], a, b))

    for node in nodes:
        v_max = node.config.u_max - node.config.u_min
        mat = node.config.material
        if node.prev is None:
            make_cap(reversed(node.points_next), -node.orient.forward())
        if node.next is None:
            make_cap(node.points_prev, node.orient.forward())


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
