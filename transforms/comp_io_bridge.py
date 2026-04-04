"""IO Bridge between instances, submaps, etc. Mainly used to send IO to/from instances in P1
"""

from hammeraddons.bsp_transform import trans, Context
from srctools.logger import get_logger
from srctools import Entity, VMF, Output


LOGGER = get_logger(__name__)



@trans("comp_io_bridge")
def comp_io_bridge(ctx: Context) -> None:
    vmf: VMF = ctx.vmf

    for bent in vmf.by_class["comp_io_bridge"]:

        if not (bent_name := bent["targetname", ""]):
            LOGGER.warning(f"Entity comp_io_bridge at {bent.get_origin()} has no targetname! Skipping...")
            continue

        targets = vmf.by_target[bent_name]

        links: set[Entity] = set()

        for target in targets:
            if target["classname"] == "comp_io_bridge" and bent != target:
                links.add(target)

        outputs_to_us: list[tuple[Entity, Output]] = [] # We can't do this via iterating over inputs search since we need the entity too
        for ent in vmf.entities:
            for output in ent.outputs:
                if output.target == bent_name:
                    outputs_to_us.append((ent, output))


        link_outputs: list[Output] = []
        for _link in links:
            link_outputs.extend(_link.outputs)

        for src_ent, inp in outputs_to_us:
            for out in link_outputs: # Output is the parent entity's out

                if not out.output == inp.input: # The input to us, like Relay1 needs to match the partner's output
                    continue
                
                output_ = inp.output
                target = out.target
                inp_to_fire = out.input
                param = out.params
                times = out.times
                delay = inp.delay + out.delay

                src_ent.add_out(
                    Output(output_, target, inp_to_fire, param, delay, times=times)
                )

            id = src_ent.outputs.index(inp)
            del src_ent.outputs[id] # Delete the processed output

    
    # Cleanup
    for ent in vmf.by_class["comp_io_bridge"]:
        vmf.remove_ent(ent)



                


        