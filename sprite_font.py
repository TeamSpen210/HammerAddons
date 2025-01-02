"""Generate entity sprite text images."""
from pathlib import Path

from PIL import Image, ImageChops
from srctools import conv_bool
from collections import namedtuple

import sys

try:
    text = sys.argv[1]
except IndexError:
    text = input('Enter text to produce: ')

golden = text.startswith("comp_")
print('Gold' if golden else 'White', 'text selected')

Char = namedtuple('Char', 'img width')

LETTERS = {}

for file in Path('text').glob('*.png'):
    letter = file.name[0]
    img = Image.open(file)
    img.load()
    if golden:
        img = ImageChops.multiply(img, Image.new('RGBA', img.size, (224, 174, 0, 255)))
    LETTERS[letter] = Char(img, img.width-1)

chars = list(map(LETTERS.__getitem__, text.lower()))

width = sum(c.width for c in chars) + 1
height = max(c.img.height for c in chars)

img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

offset = 0
for ch in chars:
    img.alpha_composite(ch.img, (offset, 0))
    offset += ch.width

img.save(text + '.png')

print('Done!')
