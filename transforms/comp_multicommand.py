"""Specifies a list of commands, which will be bundled into inputs and fired on map spawn or on trigger."""
from srctools import Entity, Output, VMF, conv_int

from hammeraddons.bsp_transform import trans, Context
from hammeraddons.bsp_transform.common import strip_cust_keys, ent_description


@trans('comp_multi_command')
def comp_multi_command(ctx: Context) -> None:
    """Implement comp_multi_command."""
    command_ents: dict[str, Entity] = {}
    for comp_ent in ctx.vmf.by_class['comp_multi_command']:
        command_caller = get_command_executor(ctx.vmf, command_ents, comp_ent)

        command_list: list[tuple[int, int, str]] = []
        for name, command in comp_ent.items():
            # Hammer allows some_kv#2, for duplicates. And we support command_4. So check
            # for both.
            if not name.startswith('command_'):
                continue
            name = name.removeprefix('command_')
            if '#' in name:
                a, b = name.split('#', 1)
                command_list.append((conv_int(a), conv_int(b), command))
            else:
                command_list.append((conv_int(name), 0, command))

        command_list.sort()

        match comp_ent['mode'].casefold():
            case 'spawn':
                comp_ent['classname'] = 'logic_auto'
                comp_ent['targetname'] = ''
                output = 'OnMapSpawn'
            case 'trigger':
                comp_ent['classname'] = 'logic_relay'
                output = 'OnTrigger'
            case _:
                raise ValueError(
                    f'Invalid comp_multi_command mode "{comp_ent['mode']}" '
                    f'for {ent_description(comp_ent)}'
                )

        buffer: list[str] = []
        buf_size = 0
        for _, _, command in command_list:
            size = len(command) + 1
            if buf_size + size >= 250:
                # Output limit hit, produce one IO.
                comp_ent.add_out(Output(
                    output, command_caller, "Command",
                    ";".join(buffer),
                ))
                buffer.clear()
            buffer.append(command)
            buf_size += size
        if buffer:
            comp_ent.add_out(Output(output, command_caller, "Command", ";".join(buffer)))
        strip_cust_keys(comp_ent)


def get_command_executor(vmf: VMF, existing: dict[str, Entity], comp_ent: Entity) -> Entity:
    """Locate a suitable point_*command entity. """
    match comp_ent["type"].casefold():
        case 'client':
            classname = 'point_clientcommand'
        case 'server':
            classname = 'point_servercommand'
        case 'multiplayer':
            classname = 'point_broadcastclientcommand'
        case _:
            raise ValueError(
                f'Invalid command entity type '
                f'"{comp_ent['type']}" for {ent_description(comp_ent)}'
            )
    try:
        return existing[classname]
    except KeyError:
        pass

    # It doesn't exist, create one.
    existing[classname] = command = vmf.create_ent(classname)
    command.make_unique(f'cmp_multi_{classname}')
    return command
