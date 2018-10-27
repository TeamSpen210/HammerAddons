"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""

from srctools.logger import get_logger

LOGGER = get_logger(__name__)
