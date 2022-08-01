"""Implement customisable vactubes for items."""
import subprocess
from collections import defaultdict
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict, List, Iterable, Optional
import math
import random

import trio

from srctools.mdl import MDL_EXTS
from srctools.smd import Mesh
import srctools.logger
from srctools import Vec, Output, conv_int

from hammeraddons.bsp_transform import trans, Context
from . import nodes, animations, objects


LOGGER = srctools.logger.get_logger(__name__)
# For culling, ignore points with normals offset more than this.
ANG_THRESHOLD = math.cos(math.radians(30))

QC_TEMPLATE = '''\
$modelname "{path}"
$surfaceprop "default"
$cdmaterials "models/props_backstage"

$body body "ref.smd"

$definebone "root" "" 0.0 0.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0  0.0 0.0 0.0
$attachment "move" "root" 0.0 0.0 0.0
'''

SEQ_TEMPLATE = '''\
$sequence {name} {{ 
    "{name}.smd" 
    fps {fps}
    snap
}}
'''


def find_closest(
    all_nodes: Iterable[Tuple[Vec, List[Tuple[Vec, nodes.Node]]]],
    node: nodes.Node,
    src_type: nodes.DestType,
) -> nodes.Node:
    """Search through all the nodes to find the one most aligned to this."""
    src_point = node.vec_point(1.0, src_type)
    src_norm = node.output_norm(src_type)

    best_node: Optional[nodes.Node] = None
    best_dist = math.inf

    # We're looking for if the point is inside the cylinder projecting out
    # of the node.
    for targ_norm, node_lst in all_nodes:
        if Vec.dot(src_norm, targ_norm) < ANG_THRESHOLD:
            continue
        for targ_point, targ in node_lst:
            if node is targ:
                continue
            # First check if we're beyond the target point
            off = (src_point - targ_point)
            dist = -off.dot(src_norm)
            # On the other side, or not better than what we've got.
            # Allow a tiny amount of overlap.
            if dist < -8.0 or dist > best_dist:
                continue
            # Now project the point onto the target's plane.
            # If inside, we've found it.
            if (off + dist * targ_norm).mag_sq() <= (64*64):
                best_node = targ
                best_dist = dist

    if best_node is None:
        raise ValueError(
            f'No destination found for {src_type.value} output of {node}! Junction at {src_point}.'
        )
    return best_node


@trans('Portal 2 Vactubes')
async def vactube_transform(ctx: Context) -> None:
    """Implements the dynamic Vactube system."""
    name_to_node: Dict[str, nodes.Node] = {}
    all_nodes: List[nodes.Node] = []

    for node in nodes.parse(ctx.vmf):
        all_nodes.append(node)
        name = node.name.casefold()
        if name:
            if name in name_to_node:
                LOGGER.warning('Duplicate node with name "{}"!', node.name)
            name_to_node[name] = node

    if not all_nodes:
        # No vactubes.
        return

    LOGGER.info('{} vactube nodes found.', len(all_nodes))
    LOGGER.debug('Nodes: {}', all_nodes)

    if ctx.studiomdl is None:
        raise ValueError(
            'Vactubes present, but no studioMDL path provided! '
            'Set the path to studiomdl.exe in srctools.vdf.'
        )

    obj_count, vac_objects, objects_code = objects.parse(ctx.vmf, ctx.pack)
    groups = set(objects_code)

    if not obj_count:
        raise ValueError(
            'Vactube nodes present, but no objects. '
            'You need to add comp_vactube_objects to your map '
            'to define the contents.'
        )

    LOGGER.info('{} vactube objects found.', obj_count)

    # Now join all the nodes to each other.
    # Tubes only have 90 degree bends, so a system should mostly be formed
    # out of about 6 different normals. So group by that.
    inputs_by_norm: Dict[
        Tuple[float, float, float],
        List[Tuple[Vec, nodes.Node]]
    ] = defaultdict(list)

    for node in all_nodes:
        # Spawners have no inputs.
        if isinstance(node, nodes.Spawner):
            node.has_input = True
        else:
            inputs_by_norm[node.input_norm().as_tuple()].append((node.vec_point(0.0), node))
        #     ctx.vmf.create_ent('prop_dynamic', model='models/editor/cone_helper.mdl', rendercolor='32 32 255', origin=node.vec_point(0), angles=node.input_norm().to_angle())
        # for out_type in node.out_types:
        #     ctx.vmf.create_ent('prop_dynamic', model='models/editor/cone_helper.mdl', rendercolor='255 32 32', origin=node.vec_point(1.0, out_type), angles=node.output_norm(out_type).to_angle())
        #     count = int(node.path_len(out_type) / 8)
        #     for i in range(1, count):
        #         ctx.vmf.create_ent('prop_dynamic', model='models/editor/axis_helper.mdl', origin=node.vec_point(i/count, out_type), angles='0 0 0')

    # with open(str(ctx.bsp.filename)[:-4] + '_vac.vmf', 'w') as f:
    #     ctx.vmf.export(f, inc_version=False)

    norm_inputs = [
        (Vec(norm), node_lst)
        for norm, node_lst in
        inputs_by_norm.items()
    ]

    sources: List[nodes.Spawner] = []

    LOGGER.info('Linking nodes...')
    for node in all_nodes:
        # Destroyers (or Droppers) have no inputs.
        if isinstance(node, nodes.Destroyer):
            continue
        for dest_type in node.out_types:
            override = node.ent[dest_type.manual_targ]
            if override:
                try:
                    target = name_to_node[override.casefold()]
                except KeyError:
                    raise ValueError(f'Unknown node target "{override}" for node {node}!')
                LOGGER.debug('Override: {} -> {}', node.name, target.name)
            else:
                target = find_closest(
                norm_inputs,
                node,
                dest_type,
            )
            node.outputs[dest_type] = target

            # Mark the node as having an input, for sanity checking purposes.
            # Note that nodes can have multiple inputs, if they're merging
            # paths.
            target.has_input = True
        if isinstance(node, nodes.Spawner):
            sources.append(node)
            if node.group not in groups:
                group_warn = (
                    f'Node {node} uses group "{node.group}", '
                    'which has no objects registered!'
                )
                if '' in groups:
                    # Fall back to ignoring the group, using the default
                    # blank one which is present.
                    LOGGER.warning("{} Using blank group.", group_warn)
                    node.group = ""
                else:
                    raise ValueError(group_warn)

    # Run through them again, check to see if any miss inputs.
    for node in all_nodes:
        if not node.has_input:
            raise ValueError(
                'No source found for junction '
                f'{node.ent["targetname"]} at ({node.origin})!'
            )

    LOGGER.info('Generating animations...')
    all_anims = animations.generate(sources)
    # Sort the animations by their start and end, so they ideally are consistent.
    all_anims.sort(key=lambda a: (a.start_node.origin, a.end_node.origin))

    anim_mdl_name = Path('maps', ctx.bsp_path.stem, f'vac_anim_{random.randrange(0xffffff):06x}.mdl')

    # Now generate the animation model.
    # First wipe the model.
    full_loc = ctx.game.path / 'models' / anim_mdl_name
    for ext in MDL_EXTS:
        try:
            full_loc.with_suffix(ext).unlink()
        except FileNotFoundError:
            pass

    with TemporaryDirectory(prefix='vactubes_') as temp_dir:
        # Make the reference mesh.
        mesh = Mesh.build_bbox('root', 'vacum_pipe', Vec(-8, -8, -8), Vec(8, 8, 8))
        # Aesthetics...
        U = {0.0: 740/1024.0, 1.0: 816/1024.0}
        V = {0.0: 1.0-825/1024.0, 1.0: 1.0-904/1024.0}
        for tri in mesh.triangles:
            for point in tri:
                point.tex_u, point.tex_v = U[point.tex_u], V[point.tex_v]
        with open(temp_dir + '/ref.smd', 'wb') as f:
            mesh.export(f)

        with open(temp_dir + '/prop.qc', 'w') as qc_file:
            qc_file.write(QC_TEMPLATE.format(path=anim_mdl_name))

            for i, anim in enumerate(all_anims):
                anim.name = anim_name = f'anim_{i:03x}'
                qc_file.write(SEQ_TEMPLATE.format(name=anim_name, fps=animations.FPS))

                with open(temp_dir + f'/{anim_name}.smd', 'wb') as f:
                    anim.mesh.export(f)

        args = [
            str(ctx.studiomdl),
            '-nop4', '-i',  # Ignore warnings.
            '-game', str(ctx.game.path),
            temp_dir + '/prop.qc',
        ]
        LOGGER.info('Compiling vactube animations with args={}...', args)
        try:
            proc = await trio.run_process(args, capture_stdout=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            LOGGER.error('Failed to compile vactube animations:\n{}', exc.stdout)
            sys.exit(1)
        else:
            LOGGER.debug('Compile log:\n{}', proc.stdout)

    # Ensure they're all packed.
    for ext in MDL_EXTS:
        try:
            f = full_loc.with_suffix(ext).open('rb')
        except FileNotFoundError:
            pass
        else:
            with f:
                ctx.pack.pack_file(Path('models', anim_mdl_name.with_suffix(ext)), data=f.read())

    LOGGER.info('Setting up vactube ents...')
    # Generate the shared template.
    ctx.vmf.create_ent(
        'prop_dynamic',
        targetname='_vactube_temp_mover',
        angles='0 270 0',
        origin='-16384 0 1024',
        model=str(Path('models', anim_mdl_name)),
        rendermode=10,
        solid=0,
        spawnflags=64 | 256,  # Use Hitboxes for Renderbox, collision disabled.
    )
    ctx.vmf.create_ent(
        'prop_dynamic_override',  # In case you use the physics model.
        targetname='_vactube_temp_visual',
        parentname='_vactube_temp_mover,move',
        origin='-16384 0 1024',
        model=nodes.CUBE_MODEL,
        solid=0,
        spawnflags=64 | 256,  # Use Hitboxes for Renderbox, collision disabled.
    )
    ctx.vmf.create_ent(
        'point_template',
        targetname='_vactube_template',
        template01='_vactube_temp_mover',
        template02='_vactube_temp_visual',
        origin='-16384 0 1024',
        spawnflags='2',  # Preserve names, remove originals.
    )

    # Group animations by their start point.
    anims_by_start: Dict[nodes.Spawner, List[animations.Animation]] = defaultdict(list)

    for anim in all_anims:
        anims_by_start[anim.start_node].append(anim)

    # And create a dict to link droppers to the animation they want.
    dropper_to_anim: Dict[nodes.Dropper, animations.Animation] = {}

    for start_node, anims in anims_by_start.items():
        spawn_maker = start_node.ent
        spawn_maker['classname'] = 'env_entity_maker'
        spawn_maker['entitytemplate'] = '_vactube_template'
        spawn_maker['angles'] = '0 0 0'
        orig_name = spawn_maker['targetname']
        spawn_maker.make_unique('_vac_maker')
        spawn_name = spawn_maker['targetname']

        if start_node.is_auto:
            spawn_timer = ctx.vmf.create_ent(
                'logic_timer',
                targetname=spawn_name + '_timer',
                origin=start_node.origin,
                startdisabled='0',
                userandomtime='1',
                lowerrandombound=start_node.time_min,
                upperrandombound=start_node.time_max,
            ).make_unique()
            spawn_timer.add_out(Output('OnTimer', spawn_name, 'CallScriptFunction', 'make_cube'))
            ctx.add_io_remap(
                orig_name,
                Output('EnableTimer', spawn_timer, 'Enable'),
                Output('DisableTimer', spawn_timer, 'Disable'),
            )
        ctx.add_io_remap(
            orig_name,
            Output('ForceSpawn', spawn_name, 'CallScriptFunction', 'make_cube'),
        )

        # Now, generate the code so the VScript knows about the animations.
        code = [f'// Node: {start_node.ent["targetname"]}, {start_node.origin}']
        for anim in anims:
            target = anim.end_node
            anim_speed = anim.start_node.speed
            pass_code = ','.join([
                f'Output({time:.2f}, "{node.ent["targetname"]}", '
                f'{node.tv_code(anim_speed)})'
                for time, node in anim.pass_points
            ])
            cube_name = 'null'
            if isinstance(target, nodes.Dropper):
                cube_model = target.cube['model'].replace('\\', '/')
                cube_skin = conv_int(target.cube['skin'])
                try:
                    cube_name = vac_objects[start_node.group, cube_model, cube_skin].id
                except KeyError:
                    LOGGER.warning(
                        'Cube model "{}", skin {} is not a type of cube travelling '
                        'in this vactube!\n\n'
                        'Add a comp_vactube_object entity with this cube model'
                        # Mention groups if they're used, otherwise it's not important.
                        + (f' with the group "{start_node.group}".' if start_node.group else '.'),
                        cube_model, cube_skin,
                    )
                    continue  # Skip this animation so it's not broken.
                else:
                    dropper_to_anim[target] = anim
            code.append(
                f'{anim.name} <- anim("{anim.name}", {anim.duration}, '
                f'{cube_name}, [{pass_code}]);'
            )
        spawn_maker['vscripts'] = ' '.join([
            'srctools/vac_anim.nut', objects_code[start_node.group],
            ctx.pack.inject_vscript('\n'.join(code)),
        ])

    # Now, go through each dropper and generate their logic.
    for dropper, anim in dropper_to_anim.items():
        # Pick the appropriate output to fire once left the dropper.
        if dropper.cube['classname'] == 'prop_monster_box':
            cube_input = 'BecomeMonster'
        else:
            cube_input = 'EnablePortalFunnel'

        ctx.add_io_remap(
            dropper.ent['targetname'],
            # Used to dissolve the existing cube when respawning.
            Output('FireCubeUser1', dropper.cube['targetname'], 'FireUser1'),
            # Tell the spawn to redirect a cube to us.
            Output(
                'RequestSpawn',
                anim.start_node.ent['targetname'],
                'RunScriptCode',
                f'{anim.name}.req_spawn = true',
            ),
            Output('CubeReleased', '!activator', cube_input),
        )
