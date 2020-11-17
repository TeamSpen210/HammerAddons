""""Compile static prop cables, instead of sprites."""
import math
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import (
    Optional, List, Tuple, NamedTuple, FrozenSet,
    TypeVar, MutableMapping, NewType, Set,
)

from srctools import Vec, Entity, conv_int, conv_float, logger
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


class InterpType(Enum):
    """Type of interpolation to use."""
    STRAIGHT = 0
    SPLINE = 1
    CATENARY = 2


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


class Node:
    """A part of the rope.

    The node ID is used to match ropes to previous compilations - it'll be the
    entity's hammerid.

    The following attributes are used for equality/hashing
        * The ID
        * The position
        * The config
    """
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
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None
        self.pos = pos
        self.norm = Vec()  # Pointing from prev to next.
        # The points for the cylinder, on these sides.
        self.points_prev: List[Vertex] = []
        self.points_next: List[Vertex] = []

    def __repr__(self) -> str:
        return f'<Node "{self.id}" @ {self.pos}>'

    def __hash__(self) -> int:
        """Allow this to be hashed."""
        return hash((
            self.id,
            self.pos.x, self.pos.y, self.pos.z,
            self.config,
        ))

    def __eq__(self, other: object) -> object:
        """Allow this to be compared."""
        if isinstance(other, Node):
            return (
                self.id == other.id and
                self.pos == other.pos and
                self.config == other.config
            )
        return NotImplemented

    def __ne__(self, other: object) -> object:
        """Allow this to be compared."""
        if isinstance(other, Node):
            return (
                self.id != other.id or
                self.pos != other.pos or
                self.config != other.config
            )
        return NotImplemented


def build_rope(
    nodes_and_conn: Tuple[FrozenSet[Node], FrozenSet[Tuple[NodeID, NodeID]]],
    temp_folder: Path,
    mdl_name: str,
) -> None:
    """Construct the geometry for a rope."""
    LOGGER.info('Building rope {}', mdl_name)
    nodes, connections = nodes_and_conn
    mesh = Mesh.blank('root')
    [bone] = mesh.bones.values()
    bone_weight = [(bone, 1.0)]

    id_to_node = {node.id: node for node in nodes}

    # First connect all the nodes.
    for id1, id2 in connections:
        first = id_to_node[id1]
        second = id_to_node[id2]
        assert first.next is None, (first, second)
        assert second.prev is None, (first, second)
        first.next = second
        second.prev = first

    for node in nodes:
        if node.next is None:
            continue
        forward = (node.next.pos - node.pos).norm()
        ang = forward.to_angle()
        u = Vec(y=1).rotate(*ang)
        v = Vec(z=1).rotate(*ang)

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


@trans('Model Ropes')
def comp_prop_rope(ctx: Context) -> None:
    """Build static props for ropes."""
    compiler = ModelCompiler.from_ctx(ctx, 'ropes')
    # group -> id -> node.
    all_nodes: MutableMapping[str, MutableMapping[NodeID, Node]] = defaultdict(dict)
    # Given a targetname, all the nodes with that name.
    name_to_ids: MutableMapping[str, List[Node]] = defaultdict(list)
    # Store the node/next-key pairs for linking after they're all parsed.
    temp_conns: List[Tuple[Node, str]] = []

    for ent in ctx.vmf.by_class['comp_prop_rope'] | ctx.vmf.by_class['comp_prop_cable']:
        ent.remove()
        conf = Config.parse(ent)
        node = Node(
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

    # TODO, compute the positions.
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
