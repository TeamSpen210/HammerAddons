"""A custom logic entity to correctly sequence portal piston platforms."""
import re

from srctools import logger, Entity

from hammeraddons.bsp_transform import Context, ent_description, trans

LOGGER = logger.get_logger(__name__)
RE_PISTON = re.compile(r'piston([0-9])(?:#([0-9]+))?')


@trans('Portal Piston Platforms')
def piston_platform(ctx: Context) -> None:
    """A custom logic entity to correctly sequence portal piston platforms."""
    for ent in ctx.vmf.by_class['comp_piston_platform']:
        generate_platform(ctx, ent)


def generate_platform(ctx: Context, logic_ent: Entity) -> None:
    """Generate a piston platform."""
    # First locate and validate all piston keyvalues.
    # Kinda complicated, we want to handle a few situations.
    # First iterate all KVs, parse the number.
    # It's possible to have KVs like "blah#8" to add multiples, so order those within
    # the normal index.
    numbered_pistons = []
    for key, ent_name in logic_ent.items():
        if (match := RE_PISTON.match(key)) is None:
            continue
        ents = list(ctx.vmf.search(ent_name))
        ind = int(match.group(1))
        sub_ind = int(match.group(2)) if match.group(2) is not None else 0
        if len(ents) == 1:
            numbered_pistons.append((ind, sub_ind, ents[0]))
        elif len(ents) > 1:
            raise ValueError(
                f'{ent_description(logic_ent)} located multiple entities '
                f'for piston segment "{key}" = "{ent_name}"!'
            )
        else:
            # Just a warning, so you can start with a prefab, remove unnecessary bits and it'll
            # still work.
            LOGGER.warning(
                '{} could not find piston segment "{}"!',
                ent_description(logic_ent), key,
            )

    numbered_pistons.sort(key=lambda t: (t[0], t[1]))
    pistons = [pist for i, j, pist in numbered_pistons]
    if not pistons:
        raise ValueError(f'{ent_description(logic_ent)} has no piston segments!')

    underside_fizz: Entity | None = None
    underside_hurt: Entity | None = None
    # Remove duplicates, then remove blanks/unset.
    for ent_name in filter(None, {
        logic_ent['underside_fizz'].casefold(),
        logic_ent['underside_hurt'].casefold(),
    }):
        for ent in ctx.vmf.search(ent_name):
            # Use a multiple to fire Break/self-destruct inputs,
            # or just a fizzler to fizzle everything.
            if ent['classname'] in ('trigger_multiple', 'trigger_portal_cleanser'):
                if underside_fizz is not None:
                    raise ValueError(
                        '{} found duplicate underside hurt triggers {} and {}!',
                        ent_description(logic_ent),
                        ent_description(underside_hurt), ent_description(ent)
                    )
                underside_fizz = ent
            if ent['classname'] == 'trigger_hurt':
                if underside_hurt is not None:
                    raise ValueError(
                        '{} found duplicate underside hurt triggers {} and {}!',
                        ent_description(logic_ent),
                        ent_description(underside_hurt), ent_description(ent)
                    )
                underside_hurt = ent
