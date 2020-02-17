"""Implement customisable vactubes for items."""
import subprocess
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple, Dict, List, Iterable, Optional
import math

from srctools.compiler.propcombine import MDL_EXTS
from srctools.smd import Mesh

import srctools.logger
from srctools import Vec, Output
from srctools.bsp_transform import trans, Context
from srctools.bsp_transform.packing import comp_precache_model
from srctools.bsp_transform.vactubes import nodes
from srctools.bsp_transform.vactubes import animations, objects


LOGGER = srctools.logger.get_logger(__name__)
# For culling, ignore points with normals offset more than this.
ANG_THRESHOLD = math.cos(math.radians(30))

QC_TEMPLATE = '''\
$modelname "{path}"
$surfaceprop "default"

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
    src_point: Vec,
    src_norm: Vec,
) -> nodes.Node:
    """Search through all the nodes to find the one most aligned to this."""
    best_node = None  # type: Optional[nodes.Node]
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
            if dist < -2.0 or dist > best_dist:
                continue
            # Now project the point onto the target's plane.
            # If inside, we've found it.
            if (off + dist * targ_norm).mag_sq() <= (64*64):
                best_node = targ
                best_dist = dist

    if best_node is None:
        if node.ent['targetname']:
            name = f'"{node.ent["targetname"]}" '
        else:
            name = ''
        raise ValueError(
            f'No destination found for '
            f'junction {name} at ({node.origin})!'
        )
    # Mark the node as having an input, for sanity checking purposes.
    best_node.has_input = True

    return best_node


@trans('Portal 2 Vactubes')
def vactube_transform(ctx: Context) -> None:
    """Implements the dynamic Vactube system."""
    all_nodes = list(nodes.parse(ctx.vmf))
    if not all_nodes:
        # No vactubes.
        return
    LOGGER.info('{} vactube nodes found.', len(all_nodes))

    if ctx.studiomdl is None:
        LOGGER.error('Vactubes present, but no studioMDL path provided!')
        return

    vac_objects, objects_code = objects.parse(ctx.vmf, ctx.pack)

    LOGGER.info('{} vactube objects found.', len(vac_objects))
    if not vac_objects:
        return  # Nothing for inside vactubes.

    # Run this again, so it grabs our new ents if not already.
    comp_precache_model(ctx)

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

    norm_inputs = [
        (Vec(norm), node_lst)
        for norm, node_lst in
        inputs_by_norm.items()
    ]

    sources = []  # type: List[nodes.Spawner]

    LOGGER.info('Linking nodes...')
    for node in all_nodes:
        # Destroyers (or Droppers) have no inputs.
        if isinstance(node, nodes.Destroyer):
            continue
        node.output = find_closest(
            norm_inputs,
            node,
            node.vec_point(1.0),
            node.output_norm(),
        )
        if isinstance(node, nodes.Splitter):
            node.output_sec = find_closest(
                norm_inputs,
                node,
                node.vec_point(1.0, sec=True),
                node.output_sec_norm(),
            )
        if isinstance(node, nodes.Spawner):
            sources.append(node)

    # Run through them again, check to see if any miss inpus.
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

    anim_mdl_name = Path('maps', ctx.bsp_path.stem, 'vac_anim.mdl')

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
        with open(temp_dir + '/ref.smd', 'wb') as f:
            Mesh.build_bbox('root', 'demo', Vec(-32, -32, -32), Vec(32, 32, 32)).export(f)

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
        LOGGER.info('Compiling vactube animations {}...', args)
        subprocess.run(args)

    # Ensure they're all packed.
    for ext in MDL_EXTS:
        try:
            f = full_loc.with_suffix(ext).open('rb')
        except FileNotFoundError:
            LOGGER.warning('No {}!', full_loc.with_suffix(ext))
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
        'prop_dynamic',
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
    dropper_to_anim = {}   # type: Dict[nodes.Dropper, animations.Animation]

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
            pass_code = ','.join([
                f'Output({time:.2f}, "{node.ent["targetname"]}", '
                f'{node.tv_name()})'
                for time, node in anim.pass_points
            ])
            cube_name = 'null'
            if isinstance(target, nodes.Dropper):
                cube_model = target.cube['model'].replace('\\', '/')
                try:
                    cube_name = vac_objects[cube_model].id
                except KeyError:
                    LOGGER.warning(
                        'Cube model "{}" doesn\'t '
                        'travel in vactubes!\n\n'
                        'Add a comp_vactube_object entity.',
                        cube_model,
                    )
                else:
                    dropper_to_anim[target] = anim
            code.append(
                f'{anim.name} <- anim("{anim.name}", {anim.duration}, '
                f'{cube_name}, [{pass_code}]);'
            )
        spawn_maker['vscripts'] = ' '.join([
            'srctools/vac_anim.nut', objects_code,
            ctx.pack.inject_vscript('\n'.join(code)),
        ])

    # Now, go through each dropper and generate their logic.
    for dropper, anim in dropper_to_anim.items():
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
            )
        )
