""""Compile static prop cables, instead of sprites."""
import itertools
import math
import struct
from random import Random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    Optional, List, Tuple, FrozenSet,
    TypeVar, MutableMapping, NewType, Set, Iterable, Dict, Iterator,
)

import attr

from srctools import (
    logger, conv_int, conv_float, conv_bool,
    Vec, Entity, Matrix, Angle, lerp, FileSystem,
)
from srctools.compiler.mdl_compiler import ModelCompiler
from srctools.bsp_transform import Context, trans
from srctools.bsp import StaticProp, StaticPropFlags, VisLeaf, VisTree
from srctools.smd import Mesh, Vertex, Triangle, Bone


LOGGER = logger.get_logger(__name__)
NodeID = NewType('NodeID', str)
Number = TypeVar('Number', int, float)

try:
    from .vactubes import nodes as vac_node_mod  # type: ignore
except ImportError:
    LOGGER.exception('No vactube transform:')
    vac_node_mod = None

QC_TEMPLATE = '''\
$staticprop
$modelname "{path}"

$body body "cable.smd"
$cdmaterials ""
$sequence idle "cable.smd" act_idle 1
$illumposition {light_origin}

$keyvalues {{
    no_propcombine 1
}}
'''

QC_TEMPLATE_PHYS = '''\
$collisionmodel "cable_phy.smd" {{
    $automass
    $concave
    $maxconvexpieces {count}
}}
'''


class InterpType(Enum):
    """Type of interpolation to use."""
    STRAIGHT = 0
    CATMULL_ROM = 1
    ROPE = 2


class RopeType(Enum):
    """Type of rope, for indicating special functionality."""
    ROPE = 'rope'
    VAC_PROP = 'vac_prop'  # Static prop
    VAC_FUNCTIONAL = 'vactube_functional'  # Also produces path for vactube system.

    @property
    def is_vactube(self) -> bool:
        """Check if this is a vactube."""
        return self._value_ != 'rope'


class SegPropOrient(Enum):
    """Type of orientation for props placed on ropes."""
    NONE = 'none'
    FULL_ROT = 'follow'
    YAW_ONLY = 'yaw'
    PITCH_YAW = 'pitch_yaw'
    RAND_YAW = 'rand_yaw'
    RAND_FULL = 'rand'


@attr.define
class RopePhys:
    """Holds the data for move_rope simulation."""
    pos: Vec
    prev_pos: Vec = attr.ib(
        init=False,
        default=attr.Factory(lambda s: s.pos.copy(), takes_self=True)
    )
    radius: float  # Just to transfer to the node.


ROPE_GRAVITY = -1500
SIM_TIME = 5.00
TIME_STEP = 1/50


@attr.frozen
class SegPropConf:
    """Defines configuration for a set of props placed across the rope."""
    weight: int
    place_interval: int  # Place every X segments.
    distance: float  # Alternatively place with this minimum distance.
    model: str
    orient: SegPropOrient
    angles: Matrix

    def __hash__(self) -> int:
        """Handle hashing the matrix."""
        return hash(
            (self.weight, self.model, self.orient, self.place_interval) +
            self.angles.to_angle().as_tuple()
        )


VAC_SEG_CONF = SegPropConf(
    weight=1,
    place_interval=0,
    distance=128.0,
    model='modelsrc/vactube_ring.smd',
    orient=SegPropOrient.FULL_ROT,
    angles=Matrix(),
)
VAC_SEG_CONF_SET = frozenset({VAC_SEG_CONF})
VAC_RADIUS = 45.0
VAC_COLL_RADIUS = 52.0
VAC_MAT = 'models/props_backstage/vacum_pipe'


@attr.define
class SegProp:
    """Definition for an actually placed segment prop."""
    model: str
    offset: Vec
    orient: Matrix


@attr.s(auto_attribs=True, frozen=True, hash=True, eq=True)
class Config:
    """Configuration specified in rope entities. This can be shared to reduce duplication."""
    type: RopeType
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
    coll_segments: int
    coll_side_count: int
    seg_props: FrozenSet[SegPropConf]
    prop_rendercolor: Tuple[float, float, float]
    prop_renderalpha: int
    prop_no_shadows: bool
    prop_no_vert_light: bool
    prop_no_self_shadow: bool
    prop_light_bounce: bool
    prop_fade_min_dist: float
    prop_fade_max_dist: float
    prop_fade_scale: float

    @staticmethod
    def _parse_min(ent: Entity, keyvalue: str, minimum: Number, message: str) -> Number:
        """Helper for passing all the numeric keys."""
        value = (conv_float if isinstance(minimum, float) else conv_int)(ent[keyvalue], minimum)
        if value < minimum:
            LOGGER.warning(message, ent['origin'])
            return minimum
        return value

    @property
    def is_vactube(self) -> bool:
        """Check if this is a vactube."""
        return self.type.is_vactube

    @classmethod
    def parse(cls, ent: Entity, name_to_segprops: Dict[str, FrozenSet[SegPropConf]]) -> 'Config':
        """Parse from an entity."""
        segments = cls._parse_min(
            ent, 'segments', 0,
            'Segment count for rope at '
            '{} must be positive or zero!'
        )

        if ent['classname'].casefold() == 'comp_vactube_spline':
            # More restricted config, most are preset.
            skin = conv_int(ent['skin'])
            rope_type = RopeType.VAC_FUNCTIONAL if skin == 1 else RopeType.VAC_PROP
            if conv_bool(ent['opaque']):
                material = 'models/props_backstage/vacum_pipe_opaque'
            else:
                material = 'models/props_backstage/vacum_pipe_glass'

            # Side counts are the same as the original models.
            side_count = 24
            if conv_bool(ent['collisions']):
                coll_side_count = 12
                coll_segments = math.ceil(segments / 2)
            else:
                coll_side_count = 0
                coll_segments = -1
            radius = VAC_RADIUS
            slack = 0  # Unused.
            interp_type = InterpType.CATMULL_ROM
            u_min = 0.0
            u_max = 1.0
            v_scale = 1.0
            flip_uv = False
            seg_props = VAC_SEG_CONF_SET
        else:
            rope_type = RopeType.ROPE
            # There's not really a vanilla material we can use for cables.
            material = ent['material']
            if not material:
                raise ValueError(f'No material for rope "{ent["targetname"]}" at {ent["origin"]}')

            side_count = cls._parse_min(
                ent, 'sides', 3,
                'Ropes cannot have less than 3 sides! (node at {})',
            )
            coll_segments = cls._parse_min(
                ent, 'coll_segments', -1,
                'Collision segment count for rope at '
                '{} must be positive or zero!'
            )
            coll_side_count = conv_int(ent['coll_sides'])
            radius = cls._parse_min(
                ent, 'radius', 0.1,
                'Radius for rope at {} must be positive!',
            )
            slack = cls._parse_min(
                ent, 'slack', 0.0,
                'Rope at {} cannot have a negative slack!',
            )
            try:
                interp_type = InterpType(int(ent['positioninterpolator', '2']))
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
            flip_uv = conv_bool(ent['mat_rotate'])

            try:
                seg_props = name_to_segprops[ent['bunting'].casefold()]
            except KeyError:
                seg_props = frozenset()

        alpha = max(0, min(255, conv_int(ent['renderamt'], 255)))
        # Rescale this, so that if it's 1, the pixels are square.
        v_scale *= (u_max - u_min) / (2*math.pi*radius)

        return cls(
            rope_type,
            material,
            segments,
            side_count,
            radius,
            interp_type,
            slack,
            u_min, u_max,
            v_scale,
            flip_uv,
            coll_segments,
            coll_side_count,
            seg_props,
            tuple(Vec.from_str(ent['rendercolor'], 255, 255, 255)),
            alpha,
            conv_bool(ent['disableshadows']),
            conv_bool(ent['disablevertexlighting']),
            conv_bool(ent['disableselfshadowing']),
            conv_bool(ent['enablelightbounce']),
            conv_float(ent['fademindist'], -1.0),
            conv_float(ent['fademaxdist'], 0.0),
            conv_float(ent['fadescale'], 0.0),
        )

    def coll(self) -> Optional['Config']:
        """Extract the collision options from the ent."""
        return attr.evolve(
            self,
            material='phy',
            segments=self.segments if self.coll_segments == -1 else self.coll_segments,
            side_count=self.coll_side_count,
            radius=VAC_COLL_RADIUS if self.type.is_vactube else self.radius,
        )


@attr.frozen
class NodeEnt:
    """A node entity, and its associated configuration. This is used to match with earlier compiles."""
    pos: Vec
    config: Config = attr.ib(repr=False)
    id: NodeID
    # Nodes with the same group compile together. But it doesn't matter for
    # comparisons.
    group: str = attr.ib(eq=False, hash=False)

    def relative_to(self, off: Vec) -> 'NodeEnt':
        """Return a copy relative to the specified origin."""
        return NodeEnt(
            self.pos - off,
            self.config,
            self.id,
            self.group,
        )

    def __hash__(self) -> int:
        """Hash the vector with the rest of the values."""
        return hash((
            self.id,
            self.pos.x, self.pos.y, self.pos.z,
            self.config,
        ))


@attr.define(eq=False)
class Node:
    """All the data for a node, used during construction of the geo.

    Unlike NodeEnt, this is compared by identity, and has no ID.
    """
    pos: Vec
    config: Config
    radius: float = attr.Factory(lambda s: s.config.radius, takes_self=True)
    prev: Optional['Node'] = None
    next: Optional['Node'] = None
    # Orientation of the segment up to the next.
    orient: Matrix = attr.Factory(Matrix)
    # The points for the cylinder, on these sides.
    points_prev: List[Vertex] = attr.Factory(list)
    points_next: List[Vertex] = attr.Factory(list)

    @classmethod
    def from_ent(cls, ent: NodeEnt) -> 'Node':
        """Construct from the data in a NodeEnt."""
        return Node(ent.pos.copy(), ent.config, ent.config.radius)

    def clone(self) -> 'Node':
        """Create a duplicate of this node, but with no connections."""
        return Node(self.pos.copy(), self.config, self.radius)

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

    def follow_no_endpoints(self) -> Iterator['Node']:
        """Iterate over the nodes between this and the end."""
        node = self.next
        while node is not None and node is not self and node.next is not None:
            yield node
            node = node.next

    def __repr__(self) -> str:
        return f'<Node at {self.pos}>'


def build_rope(
    nodes_and_conn: Tuple[FrozenSet[NodeEnt], FrozenSet[Tuple[NodeID, NodeID]]],
    temp_folder: Path,
    mdl_name: str,
    args: Tuple[Vec, FileSystem],
) -> Tuple[Vec, List[Tuple[Vec, float, Vec, float]], List[SegProp], List[List[Vec]]]:
    """Construct the geometry for a rope."""
    LOGGER.info('Building rope {}', mdl_name)
    ents, connections = nodes_and_conn
    offset, fsys = args

    mesh = Mesh.blank('root')
    coll_mesh = Mesh.blank('root')
    [bone] = mesh.bones.values()

    nodes, coll_nodes = build_node_tree(ents, connections)

    interpolate_all(nodes)
    compute_orients(nodes)
    compute_verts(nodes, bone, is_coll=False)

    mesh.triangles.extend(generate_straights(nodes))
    generate_caps(nodes, mesh, is_coll=False)

    # All or nothing.
    is_vactube = next(iter(nodes)).config.is_vactube
    vac_points: List[List[Vec]] = []
    if is_vactube:
        mesh.triangles.extend(generate_vac_beams(nodes, bone, vac_points))

    seg_props = list(place_seg_props(nodes, fsys, mesh))

    if coll_nodes:
        # Generate the collision mesh.
        interpolate_all(coll_nodes)
        compute_orients(coll_nodes)
        compute_verts(coll_nodes, bone, is_coll=True)

        coll_mesh.triangles.extend(generate_straights(coll_nodes))
        generate_caps(coll_nodes, coll_mesh, is_coll=True)

    # Move the UVs around so they don't extend too far.
    for tri in mesh.triangles:
        u = math.floor(min(point.tex_u for point in tri))
        v = math.floor(min(point.tex_v for point in tri))
        if u or v:
            tri.point1 = tri.point1.with_uv(tri.point1.tex_u - u, tri.point1.tex_v - v)
            tri.point2 = tri.point2.with_uv(tri.point2.tex_u - u, tri.point2.tex_v - v)
            tri.point3 = tri.point3.with_uv(tri.point3.tex_u - u, tri.point3.tex_v - v)

    # Use the node closest to the center. That way
    # it shouldn't be inside walls, and be about representative of
    # the whole model.
    light_origin = min((node.pos for node in nodes), key=Vec.mag_sq)

    with (temp_folder / 'cable.smd').open('wb') as fb:
        mesh.export(fb)
    if coll_nodes:
        with (temp_folder / 'cable_phy.smd').open('wb') as fb:
            coll_mesh.export(fb)

    with (temp_folder / 'model.qc').open('w') as f:
        # Desolation needs this hint.
        if is_vactube and hasattr(Mesh, 'NEED_TRANSLUCENT_MOSTLYOPAQUE'):
            f.write('$mostlyopaque\n')
        f.write(QC_TEMPLATE.format(path=mdl_name, light_origin=light_origin))
        if coll_nodes:
            f.write(QC_TEMPLATE_PHYS.format(count=sum(node.next is not None for node in coll_nodes)))

    # For visleaf computation, build a list of all the actual segments generated.
    coll_data = [
        (node.pos + offset, node.radius, node.next.pos + offset, node.next.radius)
        for node in nodes
        if node.next
    ]

    return (light_origin, coll_data, seg_props, vac_points)


def build_node_tree(
    ents: FrozenSet[NodeEnt],
    connections: FrozenSet[Tuple[NodeID, NodeID]],
) -> Tuple[Set[Node], Set[Node]]:
    """Convert the ents/connections definitions into a node tree."""
    # Convert them all into the real node objects.
    id_to_node: dict[str, tuple[Node, Optional[Node]]] = {}
    vis_nodes: set[Node] = set()
    coll_nodes: Set[Node] = set()
    for node in ents:
        vis_node = Node(node.pos.copy(), node.config)
        vis_nodes.add(vis_node)
        if node.config.coll_side_count >= 3:
            coll_node = Node(node.pos.copy(), node.config.coll())
            coll_nodes.add(coll_node)
        else:
            coll_node = None
        id_to_node[node.id] = (vis_node, coll_node)

    def maybe_split(nodes: Set[Node], node: Node, direction: str) -> Node:
        """Split nodes to ensure they only have 1 or 2 connections.

        If it has more, or multiple in one side, it will be converted
        to multiple that end at the same point.
        """
        if node not in nodes:  # Already split, create a copy and return.
            copy = node.clone()
            nodes.add(copy)
            return copy
        if getattr(node, direction) is not None:
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
        a_vis, a_coll = id_to_node[id1]
        b_vis, b_coll = id_to_node[id2]
        first = maybe_split(vis_nodes, a_vis, "next")
        second = maybe_split(vis_nodes, b_vis, "prev")

        first.next = second
        second.prev = first
        if a_coll is not None and b_coll is not None:
            first = maybe_split(coll_nodes, a_coll, "next")
            second = maybe_split(coll_nodes, b_coll, "prev")

            first.next = second
            second.prev = first

    # Sometimes that ends up creating extra copies, so discard those.
    for nodeset in [vis_nodes, coll_nodes]:
        for node in list(nodeset):
            if node.prev is None and node.next is None:
                nodeset.discard(node)

    return vis_nodes, coll_nodes


def interpolate_straight(node1: Node, node2: Node, seg_count: int) -> List[Node]:
    """Simply interpolate in a straight line."""
    diff = (node2.pos - node1.pos) / (seg_count + 1)
    return [
        Node(node1.pos + diff * i, node1.config, lerp(i, 0, seg_count+1, node1.radius, node2.radius))
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
    points: list[Node] = []
    for i in range(1, seg_count + 1):
        t = lerp(i, 0, seg_count + 1, t1, t2)
        A1 = (t1-t)/(t1-t0)*p0 + (t-t0)/(t1-t0)*p1
        A2 = (t2-t)/(t2-t1)*p1 + (t-t1)/(t2-t1)*p2
        A3 = (t3-t)/(t3-t2)*p2 + (t-t2)/(t3-t2)*p3

        B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
        B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3

        points.append(Node(
            (t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2,
            node1.config,
            lerp(i, 0, seg_count + 1, node1.radius, node2.radius),
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
        RopePhys(
            node1.pos + interp_diff * i,
            lerp(i, 0, seg_count + 2, node1.radius, node2.radius),
        )
        for i in range(0, seg_count + 2)
    ]
    springs = list(zip(points, points[1:]))

    time = 0
    step = TIME_STEP
    gravity = Vec(z=ROPE_GRAVITY) * step**2
    # Valve uses 3 iterations, but they only ever have 10 subdivisions.
    # More causes the springs to fail and start sagging, so increase the
    # iteration to compensate if you use more.
    constraint_iter = range(max(6, int(seg_count/2)))
    LOGGER.debug('Solving rope slack for {} nodes with {} spring iterations...', seg_count, constraint_iter)

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
        for _ in constraint_iter:
            for phys1, phys2 in springs:
                diff = phys1.pos - phys2.pos
                dist = diff.mag_sq()
                if dist > max_len_sqr:
                    diff *= 0.55 * (1 - (max_len / math.sqrt(dist)))
                    if phys1 is anchor1:
                        phys2.pos += diff
                    elif phys2 is anchor2:
                        phys1.pos -= diff
                    else:
                        # Move towards the middle.
                        phys1.pos -= 0.5 * diff
                        phys2.pos += 0.5 * diff

    return [
        Node(point.pos, node1.config, point.radius)
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
            new_node = Node(node.pos.copy(), node.config, node.radius)
            nodes.add(new_node)
            new_node.next = node.next
            node.next.prev = new_node
            node.next = None


def compute_orients(nodes: Iterable[Node]) -> None:
    """Compute the appropriate orientation for each node."""
    # This is based on the info at:
    # https://janakiev.com/blog/framing-parametric-curves/
    tangents: dict[Node, Vec] = {}
    all_nodes: set[Node] = set()
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
        # Start with an arbitrary roll for the first orientation.
        node1.orient = Matrix.from_angle(tanj1.to_angle())
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
                up = node1.orient.up() @ Matrix.axis_angle(b, math.degrees(phi))
                node2.orient = Matrix.from_basis(x=tanj2, z=up)
            node1 = node2


def compute_verts(nodes: Iterable[Node], bone: Bone, is_coll: bool) -> None:
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
            v_end = v_start + config.v_scale * (node2.pos - node1.pos).mag()
            # For collisions, adjust the normal so that it points away from the
            # midpoint.
            if is_coll:
                coll_off = (node2.pos - node1.pos) / 2.0
            else:
                coll_off = Vec()
            for i in range(count):
                ang = lerp(i, 0, count, 0, 2*math.pi)
                local = Vec(0, math.cos(ang), math.sin(ang))
                u = lerp(i, 0, count, config.u_min, config.u_max)
                point_1 = node1.radius * local @ node1.orient
                point_2 = node2.radius * local @ node2.orient

                node1.points_next.append(vert(node1.pos + coll_off, point_1 - coll_off, u, v_start))
                if node1 is not node2:
                    node2.points_prev.append(vert(node2.pos - coll_off, point_2 + coll_off, u, v_end))
            v_start = v_end


def generate_straights(nodes: Iterable[Node]) -> Iterator[Triangle]:
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
            left_b = node2.points_prev[i % side_count]
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

            yield Triangle(mat, left_a, right_b, left_b)
            yield Triangle(mat, left_a, right_a, right_b)


def generate_caps(nodes: Iterable[Node], mesh: Mesh, is_coll: bool) -> None:
    """Cap off any unfinished sides.

    We just use a simple fan layout.
    """
    def make_cap(orig: 'Iterable[Vertex]', norm: Vec):
        # Recompute the UVs to use the first bit of the cable.
        points = [
            Vertex(
                point.pos, (point.norm if is_coll else norm),
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
        if node.config.is_vactube:  # No caps on vactubes.
            continue
        if node.prev is None:
            make_cap(reversed(node.points_next), -node.orient.forward())
        if node.next is None:
            make_cap(node.points_prev, node.orient.forward())


def generate_vac_beams(nodes: Iterable[Node], bone: Bone, vac_points: List[List[Vec]]) -> Iterator[Triangle]:
    """Generate the 4 beams surrounding vactubes.

    Also save off the vactube points for functional tubes.
    """
    bone_weight = [(bone, 1.0)]
    todo = set(nodes)
    length_scale = 1 / (2*math.pi*VAC_RADIUS)
    rand = Random()
    # From original model, the V positions in the texture.
    VERT_START = 0.260
    VERT_MID = 0.626
    VERT_END = 0.992
    # For the U axis, there's 4 beam texture sets arranged identically,
    # but we only use the first 3, the 4th is for straight_b.
    BEAMS = [
        (0.008684, Matrix.from_roll(0)),
        (0.214734, Matrix.from_roll(90)),
        (0.008684, Matrix.from_roll(180)),
        (0.415821, Matrix.from_roll(270)),
    ]
    BEAM_IN = 39.3218
    BEAM_OUT = 51.75
    BEAM_WID = 2.17316
    node_pos: Vec

    def vert_node1(y: float, z: float, norm: tuple[float, float, float], u: float) -> Vertex:
        """Helper for generating at the first node."""
        return Vertex(
            pos1 + (0.0, y, z) @ orient1,
            norm @ orient1,
            u_off + u, v_start,
            bone_weight,
        )

    def vert_node2(y: float, z: float, norm: Tuple[float, float, float], u: float) -> Vertex:
        """Helper for generating a vert at the second node."""
        return Vertex(
            pos2 + (0.0, y, z) @ orient2,
            norm @ orient2,
            u_off + u, v_end,
            bone_weight,
        )

    while todo:
        start = todo.pop()
        start = start.find_start()
        if start.next is None or not start.config.is_vactube:
            continue

        points: Optional[List[Vec]]
        if start.config.type is RopeType.VAC_FUNCTIONAL:
            points = []
            vac_points.append(points)
        else:
            points = None

        v_start = VERT_START
        for node1 in start.follow():
            todo.discard(node1)

            if points is not None:
                points.append(node1.pos)
            if node1.next is None:
                continue
            node2 = node1.next

            pos1 = node1.pos
            pos2 = node2.pos

            if v_start > VERT_MID:
                v_start = VERT_START
            v_end = v_start + length_scale * (pos2 - pos1).mag()
            if v_end > VERT_END:
                v_end -= v_start - VERT_START
                v_start = VERT_START

            for u_off, orient in BEAMS:
                orient1 = orient @ node1.orient
                orient2 = orient @ node2.orient
                # Constructing a beam on the +Y side of the model, with pipe along X axis.

                # +Z side (left side)
                yield Triangle(
                    VAC_MAT,
                    vert_node1(BEAM_IN, +BEAM_WID, (0, 0, 1), 0.0),
                    vert_node2(BEAM_IN, +BEAM_WID, (0, 0, 1), 0.0),
                    vert_node1(BEAM_OUT, +BEAM_WID, (0, 0, 1), 0.07),
                )
                yield Triangle(
                    VAC_MAT,
                    vert_node2(BEAM_IN, +BEAM_WID, (0, 0, 1), 0.0),
                    vert_node2(BEAM_OUT, +BEAM_WID, (0, 0, 1), 0.07),
                    vert_node1(BEAM_OUT, +BEAM_WID, (0, 0, 1), 0.07),
                )
                # +Y (outside)
                yield Triangle(
                    VAC_MAT,
                    vert_node2(BEAM_OUT, +BEAM_WID, (0, 1, 0), 0.07),
                    vert_node2(BEAM_OUT, -BEAM_WID, (0, 1, 0), 0.095),
                    vert_node1(BEAM_OUT, +BEAM_WID, (0, 1, 0), 0.07),
                )
                yield Triangle(
                    VAC_MAT,
                    vert_node1(BEAM_OUT, +BEAM_WID, (0, 1, 0), 0.07),
                    vert_node2(BEAM_OUT, -BEAM_WID, (0, 1, 0), 0.095),
                    vert_node1(BEAM_OUT, -BEAM_WID, (0, 1, 0), 0.095),
                )
                # -Z (right side)
                yield Triangle(
                    VAC_MAT,
                    vert_node1(BEAM_OUT, -BEAM_WID, (0, 0, -1), 0.095),
                    vert_node2(BEAM_IN, -BEAM_WID, (0, 0, -1), 0.166),
                    vert_node1(BEAM_IN, -BEAM_WID, (0, 0, -1), 0.166),
                )
                yield Triangle(
                    VAC_MAT,
                    vert_node1(BEAM_OUT, -BEAM_WID, (0, 0, -1), 0.095),
                    vert_node2(BEAM_OUT, -BEAM_WID, (0, 0, -1), 0.095),
                    vert_node2(BEAM_IN, -BEAM_WID, (0, 0, -1), 0.166),
                )
                # -Y side (inner)
                yield Triangle(
                    VAC_MAT,
                    vert_node1(BEAM_IN, -BEAM_WID, (0, -1, 0), 0.166),
                    vert_node2(BEAM_IN, -BEAM_WID, (0, -1, 0), 0.166),
                    vert_node1(BEAM_IN, +BEAM_WID, (0, -1, 0), 0.191),
                )
                yield Triangle(
                    VAC_MAT,
                    vert_node2(BEAM_IN, -BEAM_WID, (0, -1, 0), 0.166),
                    vert_node2(BEAM_IN, +BEAM_WID, (0, -1, 0), 0.191),
                    vert_node1(BEAM_IN, +BEAM_WID, (0, -1, 0), 0.191),
                )
            v_start = v_end


def place_seg_props(nodes: Iterable[Node], fsys: FileSystem, mesh: Mesh) -> Iterator[SegProp]:
    """Place segment props, across the nodes."""
    mesh_cache: dict[str, Mesh] = {}
    prop_dists: dict[SegPropConf, float] = {}
    for start_node in nodes:
        # Find start nodes, we then loop in order over the nodes.
        if start_node.prev is not None:
            continue
        prop_dists.clear()
        for i, node in enumerate(start_node.follow_no_endpoints()):
            weights: List[SegPropConf] = []
            dist = (node.pos - node.next.pos).mag()
            for conf in node.config.seg_props:
                if conf.distance:
                    if prop_dists.setdefault(conf, 0.0) > conf.distance:
                        prop_dists[conf] -= conf.distance
                        weights.extend(itertools.repeat(conf, conf.weight))
                    else:
                        prop_dists[conf] += dist
                elif i % conf.place_interval == 0:
                    weights.extend(itertools.repeat(conf, conf.weight))

            if not weights:
                # None to place here, skip.
                continue
            rand = Random(struct.pack(
                '6f',
                *node.pos,
                *node.orient.forward(),
            ))

            conf = rand.choice(weights)
            if conf.orient is SegPropOrient.RAND_FULL:
                # We cover all orientations, so pre-rotation value is irrelevant.
                angles = Matrix.from_angle(Angle(
                    rand.uniform(0.0, 360.0),
                    rand.uniform(0.0, 360.0),
                    rand.uniform(0.0, 360.0),
                ))
            elif conf.orient is SegPropOrient.NONE:
                angles = conf.angles
            elif conf.orient is SegPropOrient.FULL_ROT:
                angles = conf.angles @ node.orient
            elif conf.orient is SegPropOrient.YAW_ONLY:
                angles = conf.angles @ Matrix.from_yaw(
                    node.orient.forward().to_angle().yaw,
                )
            elif conf.orient is SegPropOrient.PITCH_YAW:
                forward_ang = node.orient.forward().to_angle()
                forward_ang.roll = 0
                angles = conf.angles @ forward_ang
            elif conf.orient is SegPropOrient.RAND_YAW:
                angles = conf.angles @ Matrix.from_yaw(rand.uniform(0, 360.0))
            else:
                raise AssertionError(f'Unknown orient type {conf.orient!r}')

            folded_model = conf.model.casefold()
            if folded_model.endswith('.mdl'):
                yield SegProp(conf.model, node.pos, angles)
                continue
            try:
                prop_mesh = mesh_cache[folded_model]
            except KeyError:
                LOGGER.info('Parsing bunting mesh "{}"', conf.model)
                with fsys[conf.model].open_bin() as f:
                    prop_mesh = mesh_cache[folded_model] = Mesh.parse_smd(f)
            mesh.append_model(prop_mesh, angles, node.pos)


def compute_visleafs(
    coll_data: List[Tuple[Vec, float, Vec, float]],
    vis_tree_top: VisTree,
) -> List[VisLeaf]:
    """Compute the visleafs this rope is present in."""
    # Each tree node defines a plane. For each side we touch, we need to
    # continue looking down that side of the tree for visleafs.
    # We need to do this individually for each segment pair. That way
    # we correctly handle cases like ropes encircling a room without entering it.
    used_leafs: Set[VisLeaf] = set()

    # Check if we collide with either side of the tree (or both).
    # This just involves doing a sphere-plane check with each side of the node.
    # If both are on one side, the whole segment cannot cross.
    for point1, radius1, point2, radius2 in coll_data:
        todo_trees: List[VisTree] = [vis_tree_top]
        for tree in todo_trees:
            off1 = Vec.dot(tree.plane.normal, point1) - tree.plane.dist
            off2 = Vec.dot(tree.plane.normal, point2) - tree.plane.dist
            if off1 >= -radius1 or off2 >= -radius2:
                if isinstance(tree.child_neg, VisLeaf):
                    used_leafs.add(tree.child_neg)
                else:
                    todo_trees.append(tree.child_neg)
            if off1 <= radius1 or off2 <= radius2:
                if isinstance(tree.child_pos, VisLeaf):
                    used_leafs.add(tree.child_pos)
                else:
                    todo_trees.append(tree.child_pos)

    return list(used_leafs)


@trans('Model Ropes', priority=-10)  # Needs to be before vactubes.
def comp_prop_rope(ctx: Context) -> None:
    """Build static props for ropes."""
    # id -> node.
    all_nodes: MutableMapping[NodeID, NodeEnt] = {}
    # Given a targetname, all the nodes with that name.
    name_to_nodes: MutableMapping[str, List[NodeEnt]] = defaultdict(list)
    # Group name -> nodes with that group.
    group_to_node: Dict[str, List[NodeEnt]] = defaultdict(list)
    # Store the node/next-key pairs for linking after they're all parsed.
    temp_conns: List[Tuple[NodeEnt, str]] = []
    # Dynamic ents which will be given the static props.
    group_dyn_ents: Dict[str, List[Entity]] = defaultdict(list)
    # Name -> segprop configurations.
    name_to_segprops_lst: dict[str, list[SegPropConf]] = defaultdict(list)

    for ent in ctx.vmf.by_class['comp_prop_rope_bunting']:
        ent.remove()
        name_to_segprops_lst[ent['targetname'].casefold()].append(SegPropConf(
            max(1, conv_int(ent['weight'], 1)),
            max(1, conv_int(ent['placement_interval'], 1)),
            0.0,
            ent['model'],
            SegPropOrient(ent['orient']),
            Matrix.from_angle(Angle.from_str(ent['angles'])),
        ))

    # Put into a set, so they're immutable and have no ordering.
    # We use that to identify the same config in previous compiles.
    name_to_segprops_set: dict[str, frozenset[SegPropConf]] = {
        name: frozenset(lst)
        for name, lst in name_to_segprops_lst.items()
    }

    for ent in itertools.chain(
        ctx.vmf.by_class['comp_prop_rope'],
        ctx.vmf.by_class['comp_prop_cable'],
        ctx.vmf.by_class['comp_vactube_spline'],
    ):
        ent.remove()
        conf = Config.parse(ent, name_to_segprops_set)
        node = NodeEnt(
            Vec.from_str(ent['origin']),
            conf,
            NodeID(ent['hammerid']),
            ent['group'].casefold(),
        )
        all_nodes[node.id] = node

        if node.group:
            group_to_node[node.group].append(node)
        if ent['targetname']:
            name_to_nodes[ent['targetname'].casefold()].append(node)
        if ent['nextkey']:
            temp_conns.append((node, ent['nextkey'].casefold()))

    for ent in ctx.vmf.by_class['comp_prop_rope_dynamic'] | ctx.vmf.by_class['comp_prop_cable_dynamic']:
        ent['classname'] = 'prop_dynamic'
        group_name = ent['group']
        del ent['group']
        if group_name not in group_to_node:
            if ent['targetname']:
                LOGGER.warning('Dynamic rope "{}" has no nodes in group {}!', ent['targetname'], group_name)
            else:
                LOGGER.warning('Dynamic rope at ({}) has no nodes in group {}!', ent['origin'], group_name)
            ent.remove()
            continue
        group_dyn_ents[group_name].append(ent)

    if not all_nodes:
        return
    LOGGER.info('{} rope nodes found.', len(all_nodes))
    if ctx.studiomdl is None:
        LOGGER.warning('Ropes cannot be compiled, no StudioMDL.exe found!')
        return

    connections_to: Dict[NodeID, List[NodeEnt]] = defaultdict(list)
    connections_from: Dict[NodeID, List[NodeEnt]] = defaultdict(list)

    for node, target in temp_conns:
        found: List[NodeEnt] = []
        if target.endswith('*'):
            search = target[:-1]
            for name, nodes in name_to_nodes.items():
                if name.startswith(search):
                    found.extend(nodes)
        else:
            found.extend(name_to_nodes.get(target, ()))
        for dest in found:
            connections_from[node.id].append(dest)
            connections_to[dest.id].append(node)

    # To group nodes, take each group out, then search recursively through
    # all connections from it to other nodes.
    todo = set(all_nodes.values())
    with ModelCompiler.from_ctx(ctx, 'ropes', version=2) as compiler:
        while todo:
            dyn_ents: List[Entity] = []
            node = todo.pop()
            connections: Set[Tuple[NodeID, NodeID]] = set()
            # We need the set for fast is-in checks, and the list
            # so we can loop through while modifying it.
            nodes: Set[NodeEnt] = {node}
            unchecked: List[NodeEnt] = [node]
            while unchecked:
                node = unchecked.pop()
                # Three links to others - connections to/from, and groups.
                # We'll only ever follow a path once, so pop from the dicts.
                if node.group:
                    dyn_ents.extend(group_dyn_ents[node.group])
                    for subnode in group_to_node.pop(node.group, ()):
                        if subnode not in nodes:
                            nodes.add(subnode)
                            unchecked.append(subnode)
                for conn_node in connections_from.pop(node.id, ()):
                    connections.add((node.id, conn_node.id))
                    if conn_node not in nodes:
                        nodes.add(conn_node)
                        unchecked.append(conn_node)
                for conn_node in connections_to.pop(node.id, ()):
                    connections.add((conn_node.id, node.id))
                    if conn_node not in nodes:
                        nodes.add(conn_node)
                        unchecked.append(conn_node)
            todo -= nodes
            if len(nodes) == 1:
                LOGGER.warning('Node at {} has no connections to it! Skipping.', node.pos)
                continue

            for ent in dyn_ents:
                origin = Vec.from_str(ent['origin'])
                dyn_nodes = frozenset({
                    node.relative_to(origin)
                    for node in nodes
                })
                model_name, _ = compiler.get_model(
                    (dyn_nodes, frozenset(connections)),
                    build_rope,
                    (origin, ctx.pack.fsys),
                )
                ent['model'] = model_name
                ang = Angle.from_str(ent['angles'])
                ang.yaw -= 90.0
                ent['angles'] = ang

            if not dyn_ents:  # Static prop.
                bbox_min, bbox_max = Vec.bbox(node.pos for node in nodes)
                center = (bbox_min + bbox_max) / 2
                node = None
                has_coll = False
                local_nodes: set[NodeEnt] = set()
                for node in nodes:
                    local_nodes.add(attr.evolve(node, pos=node.pos - center))
                    if node.config.coll_side_count >= 3:
                        has_coll = True

                model_name, (light_origin, coll_data, seg_props, vac_points) = compiler.get_model(
                    (frozenset(local_nodes), frozenset(connections)),
                    build_rope,
                    (center, ctx.pack.fsys),
                )

                if vac_points and vac_node_mod is not None:
                    for track in vac_points:
                        vac_node_mod.SPLINES.append(vac_node_mod.Spline(center, track))

                # Compute the flags. Just pick a random node, from above.
                conf = node.config
                flags = StaticPropFlags.NONE
                if conf.prop_light_bounce:
                    flags |= StaticPropFlags.BOUNCED_LIGHTING
                if conf.prop_no_shadows:
                    flags |= StaticPropFlags.NO_SHADOW
                if conf.prop_no_vert_light:
                    flags |= StaticPropFlags.NO_PER_VERTEX_LIGHTING
                if conf.prop_no_self_shadow:
                    flags |= StaticPropFlags.NO_SELF_SHADOWING

                leafs = compute_visleafs(coll_data, ctx.bsp.vis_tree())
                ctx.bsp.props.append(StaticProp(
                    model=model_name,
                    origin=center,
                    angles=Angle(0, 270, 0),
                    scaling=1.0,
                    visleafs=leafs,
                    solidity=6 if has_coll else 0,
                    flags=flags,
                    tint=Vec(conf.prop_rendercolor),
                    renderfx=conf.prop_renderalpha,
                    lighting=center + light_origin,
                    min_fade=conf.prop_fade_min_dist,
                    max_fade=conf.prop_fade_max_dist,
                    fade_scale=conf.prop_fade_scale,
                ))
                for seg_prop in seg_props:
                    ctx.bsp.props.append(StaticProp(
                        model=seg_prop.model,
                        origin=center + seg_prop.offset,
                        angles=(seg_prop.orient @ Matrix.from_yaw(270)).to_angle(),
                        scaling=1.0,
                        visleafs=leafs,  # TODO: compute individual leafs here?
                        solidity=6,
                        flags=flags,
                        tint=Vec(conf.prop_rendercolor),
                        renderfx=conf.prop_renderalpha,
                        lighting=center + seg_prop.offset,
                        min_fade=conf.prop_fade_min_dist,
                        max_fade=conf.prop_fade_max_dist,
                        fade_scale=conf.prop_fade_scale,
                    ))
    LOGGER.info('Built {} models.', len(all_nodes))
