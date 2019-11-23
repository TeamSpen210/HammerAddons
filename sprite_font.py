"""Generate entity sprite text images."""
from pathlib import Path

from PIL import Image
from collections import namedtuple

import sys

Char = namedtuple('Char', 'img width')

LETTERS = {}

for file in Path('text').glob('*.png'):
    letter = file.name[0]
    img = Image.open(file)
    img.load()
    LETTERS[letter] = Char(img, img.width-1)

try:
    text = sys.argv[1]
except IndexError:
    text = input('Enter text to produce: ')

chars = list(map(LETTERS.__getitem__, text.lower()))

width = sum(c.width for c in chars) + 1

img = Image.new('RGBA', (width, 11), (0, 0, 0, 0))

offset = 0
for ch in chars:
    img.alpha_composite(ch.img, (offset, 0))
    offset += ch.width

img.save(text + '.png')

print('Done!')
