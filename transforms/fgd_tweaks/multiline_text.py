"""Allow using /n to produce newlines in game_text entities."""
from hammeraddons.bsp_transform import Context, trans


@trans('Multiline game_text')
def multiline_text(ctx: Context) -> None:
    """Allow using /n to produce newlines in game_text entities.

    Hammer converts \\ to / when saving VMFs, so backslashes don't work.
    """
    text_ents = set()
    for ent in ctx.vmf.by_class['game_text'] | ctx.vmf.by_class['game_text_tf']:
        text_ents.add(ent['targetname'].casefold())
        ent['message'] = ent['message'].replace('/n', '\n')
    for ent in ctx.vmf.entities:
        for out in ent.outputs:
            if out.input.casefold() != 'settext':
                continue
            if out.target.endswith('*'):
                search = out.target[:-1].casefold()
                if all(not name.startswith(search) for name in text_ents):
                    continue
            # Could also check for classname match, but that's fairly useless.
            elif out.target.casefold() not in text_ents:
                continue
            out.params = out.params.replace('/n', '\n')
