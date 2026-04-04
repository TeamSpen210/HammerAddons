"""Generate entity sprite text images."""
from pathlib import Path
import sys

from PIL import Image, ImageChops
import attrs


@attrs.frozen
class Char:
    """A character."""
    img: Image.Image
    width: int


def main() -> None:
    """Generate text."""
    try:
        text = sys.argv[1]
    except IndexError:
        text = input('Enter text to produce, prefix with GOLD to make it gold: ')

    golden = text.startswith("comp_")
    if text.startswith('GOLD'):
        golden = True
        text = text.removeprefix('GOLD').lstrip()
    print('Gold' if golden else 'White', 'text selected')

    LETTERS: dict[str, Char] = {}

    for file in Path('text').glob('*.png'):
        letter = file.name[0]
        img = Image.open(file)
        img.load()
        if golden:
            img = ImageChops.multiply(img, Image.new('RGBA', img.size, (224, 174, 0, 255)))
        LETTERS[letter] = Char(img, img.width-1)

    chars = [LETTERS[x] for x in text.lower()]

    width = sum(c.width for c in chars) + 1
    height = max(c.img.height for c in chars)

    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    offset = 0
    for ch in chars:
        img.alpha_composite(ch.img, (offset, 0))
        offset += ch.width

    img.save(f'{text}.png')

    print('Done!')
