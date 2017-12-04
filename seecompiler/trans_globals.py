"""Apply transformations that work on (almost) all entities."""

"""Implements various brush entities."""
from seecompiler.transformation import trans, Context
from seecompiler.logger import get_logger
from seecompiler.packlist import pack_file, FileType

LOGGER = get_logger(__name__)


@trans('Attachment Points')
def att_points(ctx: Context):
    """Allow setting attachment points in a separate field to the parent name."""
    for ent in ctx.vmf.entities:
        if not ent['parent_attachment_point']:
            continue
        parent = ent['parentname'].rsplit(',', 1)[0]

        if not parent:
            LOGGER.warning(
                'No parent, but attachment point set for "{}"? ({})',
                ent['targetname'],
                ent['origin'],
            )
            continue

        ent['parentname'] = parent + ',' + ent['parent_attachment_point']
