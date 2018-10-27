"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import re
from pathlib import Path
from typing import Dict, List, Iterator, Optional

from srctools.tokenizer import Tokenizer, Token

from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, BSP_LUMPS, StaticProp, StaticPropFlags
from srctools.mdl import Model
from collections import defaultdict, namedtuple


LOGGER = get_logger(__name__)

QC = namedtuple('QC', [
    'path',  # QC path.
    'ref_smd',    # Location of main visible geometry.
    'phy_smd',    # Relative location of collision model, or None
    'ref_scale',  # Scale of main model.
    'phy_scale',  # Scale of collision model.
])


class DynamicModel(Exception):
    """Used as flow control."""


def load_qcs(game: Game) -> Dict[str, QC]:
    """Parse through all the QC files to match to compiled models."""
    # If gameinfo is blah/game/hl2/gameinfo.txt,
    # QCs should be in blah/content/....

    qc_map = {}

    content_path = game.path.parent.parent / 'content'
    for qc_path in content_path.rglob('*.qc'):  # type: Path
        model_name = ref_smd = phy_smd = None
        scale_factor = ref_scale = phy_scale = 1.0
        qc_loc = qc_path.parent
        try:
            with open(qc_path) as f:
                tok = Tokenizer(f, qc_path, allow_escapes=False)
                for token_type, token_value in tok:

                    if model_name and ref_smd and phy_smd:
                        break

                    if token_type is Token.STRING:
                        token_value = token_value.casefold()
                        if token_value == '$scale':
                            scale_factor = float(tok.expect(Token.STRING))
                        elif token_value == '$modelname':
                            model_name = tok.expect(Token.STRING)
                        elif token_value == "$bodygroup":
                            tok.expect(Token.STRING)  # group name.
                            tok.expect(Token.BRACE_OPEN)
                            for body_type, body_value in tok:
                                if body_type is Token.BRACE_CLOSE:
                                    break
                                elif body_type is Token.STRING:
                                    if body_value.casefold() == "studio":
                                        if ref_smd:
                                            raise DynamicModel
                                        else:
                                            ref_smd = qc_loc / tok.expect(Token.STRING)
                                            ref_scale = scale_factor
                                elif body_type is not Token.NEWLINE:
                                    raise tok.error(body_type)

                        elif token_value in '$collisionmodel':
                            phy_smd = qc_loc / tok.expect(Token.STRING)
                            phy_scale = scale_factor

                        # We can't support this.
                        elif token_value in (
                            '$collisionjoints',
                            '$ikchain',
                            '$weightlist',
                            '$poseparameter',
                            '$proceduralbones',
                            '$lod',
                            '$jigglebone',
                            '$keyvalues',
                        ):
                            raise DynamicModel

        except DynamicModel:
            # It's a dynamic QC, we can't combine.
            continue
        if model_name is None or ref_smd is None:
            # Malformed...
            continue

        qc_map[model_name.casefold()] = QC(
            str(qc_path).replace('\\', '/'),
            str(ref_smd).replace('\\', '/'),
            str(phy_smd).replace('\\', '/'),
            ref_scale,
            phy_scale,
        )
    return qc_map
def combine(
    bsp: BSP,
    pack: PackList,
    game: Game,
):
    """Combine props in this map.

    temp_path is a location to copy
    """
    # Parse through all the QC files.
    LOGGER.info('Parsing QC files...')
    qc_map = load_qcs(game)
    LOGGER.info('Done! {} props.', len(qc_map))
