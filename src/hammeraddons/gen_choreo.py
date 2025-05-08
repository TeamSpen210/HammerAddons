"""Generate VCD choreo scripts for each character sound."""
from typing import Protocol

from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections import defaultdict
import argparse
import io
import pathlib
import re
import sys
import traceback

from srctools import Keyvalues, choreo, sndscript
from srctools.tokenizer import Tokenizer
from pyglet.util import DecodeException as PygletDecode
import attrs
import pyglet.media
import trio
import pyperclip


COMMENT = '// This choreo scene was auto-generated. Remove this comment if modifying this file.\n'

FILES: list[str] = []

CHOREO_SOUNDS: dict[str, dict[str, sndscript.Sound]] = defaultdict(dict)
CHOREO_TO_SOUNDSCRIPT = str.maketrans({
    '/': '.',
    '\\': '.',
    '-': '_',
})
SCENES: list[choreo.Entry] = []
RE_CAPTION_CMD = re.compile(r'<[^>]+>')
RE_WORDS = re.compile(r'\S+')
MANUAL_SOUNDSCRIPTS: set[str] = set()


@attrs.frozen
class Overridable[T]:
    """A value with a per-actor default that can be overridden per-line."""
    # Character (or ent for soundlevels) -> value
    defaults: Mapping[str, T]
    # Soundscript -> value
    overrides: Mapping[str, T]
    # Soundscript prefixes -> value
    prefixes: Sequence[tuple[str, T]]

    @classmethod
    def parse(
        cls,
        conf: Keyvalues,
        parse: Callable[[str], T],
        block_name: str, override_name: str,
    ) -> 'Overridable[T]':
        """Parse from the config."""
        defaults = {}
        for kv in conf.find_children(block_name):
            try:
                defaults[kv.real_name] = parse(kv.value)
            except (LookupError, TypeError, ValueError) as exc:
                raise ValueError(f'Could not parse {block_name} option for {kv.real_name!r}') from exc
        overrides = {}
        prefixes = []
        for kv in conf.find_children(override_name):
            # Flip, so we can group similar overrides together and it'll look nice.
            try:
                value = parse(kv.real_name)
            except (LookupError, TypeError, ValueError) as exc:
                raise ValueError(f'Could not parse {override_name} option for {kv.value!r}') from exc
            if kv.value.endswith('*'):
                prefixes.append((kv.value.removesuffix('*'), value))
            else:
                overrides[kv.value] = value
        # Sort longer first, so they override less specific ones.
        prefixes.sort(key=lambda tup: len(tup[0]), reverse=True)
        return cls(defaults, overrides, prefixes)

    def get(self, soundscript: str, character: str, default: T | None = None) -> T:
        """Get the value for this soundscript."""
        snd_fold = soundscript.casefold()
        try:
            return self.overrides[snd_fold]
        except KeyError:
            pass
        for prefix, value in self.prefixes:
            if snd_fold.startswith(prefix):
                return value

        try:
            return self.defaults[character]
        except KeyError:
            if default is not None:
                return default
            raise ValueError(
                f'Unknown character "{character}" '
                f'for soundscript "{soundscript}"!'
            ) from None


@attrs.define(kw_only=True)
class Settings:
    """General configuration."""
    game_dir: trio.Path
    # Actor entity names, or "" to skip.
    actor_names: Overridable[str]
    # Mixgroup to use, or "" to disable.
    mixgroups: Overridable[str]
    # Soundlevel to use. Looked up from actor ent, not name.
    soundlevels: Overridable[
        tuple[sndscript.Level | float, sndscript.Level | float]
        | sndscript.Level | float
    ]
    # For subtitle -> choreo, the WPM to use.
    seconds_per_word: float
    # Lists of scenes.image files to merge into ours, with the filenames to add, or None for all
    scene_imports: Mapping[trio.Path, list[str] | None]
    # Whether stacks are allowed.
    use_operator_stacks: bool
    # Overrides the captions used for specific soundscripts.
    caption_override: Mapping[str, str]


async def scene_from_sound(settings: Settings, root: trio.Path, filename: trio.Path) -> None:
    """Build a soundscript and scene for a sound."""
    try:
        duration = (await trio.to_thread.run_sync(pyglet.media.load, str(filename))).duration
    except PygletDecode as exc:
        print(f'Could not determine duration for WAV {filename}:')
        traceback.print_exception(type(exc), exc, None)
        return

    relative = filename.relative_to(root)

    scene_name = await trio.Path(
        settings.game_dir, 'scenes', 'npc', relative,
    ).with_suffix('.vcd').resolve()
    soundscript_name = str(relative.with_suffix('')).translate(CHOREO_TO_SOUNDSCRIPT).lower()

    if soundscript_name.casefold() in MANUAL_SOUNDSCRIPTS:
        return

    character = relative.parts[0]

    if settings.actor_names.get(soundscript_name, character) == "":
        return

    mixgroup = settings.mixgroups.get(soundscript_name, character, f'{character}VO')
    soundlevel = settings.soundlevels.get(soundscript_name, character, sndscript.Level.SNDLVL_NONE)

    # print(filename, '->', soundscript_name, scene_name, duration)
    CHOREO_SOUNDS[character][soundscript_name] = snd = sndscript.Sound(
        name=soundscript_name,
        sounds=[str(trio.Path('npc', relative)).replace('\\', '/')],
        channel=sndscript.Channel.VOICE,
        level=soundlevel,
    )
    if settings.use_operator_stacks and mixgroup:
        snd.stack_update = Keyvalues("update_stack", [
            Keyvalues("import_stack", "update_dialog"),
            Keyvalues("mixer", [
                Keyvalues("mixgroup", mixgroup),
            ])
        ])

    await build_scene(settings, scene_name, character, duration, soundscript_name, cc_only=False)


async def scene_from_subtitle(settings: Settings, soundscript: str, caption: str) -> None:
    """Create temporary choreo scenes from captions."""
    if '.' not in soundscript:
        return

    character, _ = soundscript.split('.', 1)
    if soundscript in CHOREO_SOUNDS[character] or soundscript.casefold() in MANUAL_SOUNDSCRIPTS:
        # Already done.
        return

    actor_name = settings.actor_names.get(soundscript, character)
    if actor_name == "":
        return

    soundlevel = settings.soundlevels.get(soundscript, actor_name, sndscript.Level.SNDLVL_NONE)

    CHOREO_SOUNDS[character][soundscript] = snd = sndscript.Sound(
        name=soundscript,
        sounds=["common/null.wav"],
        channel=sndscript.Channel.VOICE,
        level=soundlevel,
    )
    if settings.use_operator_stacks:
        snd.stack_start = Keyvalues("start_stack", [
            Keyvalues("import_stack", "P2_null_start"),
        ])

    stripped = RE_CAPTION_CMD.sub('', caption)

    word_count = sum(1 for _ in RE_WORDS.finditer(stripped))
    duration = settings.seconds_per_word * word_count
    # print(f'Caption: {soundscript} = {word_count} words = {duration}')
    scene_name = trio.Path(
        settings.game_dir, 'scenes', 'npc', *soundscript.split('.')
    ).with_suffix('.vcd')
    await build_scene(
        settings, scene_name, character, duration, soundscript,
        cc_only=True,
    )


async def build_scene(
    settings: Settings,
    scene_name: trio.Path,
    character: str,
    duration: float,
    soundscript_name: str,
    cc_only: bool,
) -> None:
    """Write a scene to a specific path."""
    actor_name = settings.actor_names.get(soundscript_name, character)
    if not actor_name:
        return

    await scene_name.parent.mkdir(parents=True, exist_ok=True)

    caption = settings.caption_override.get(soundscript_name, soundscript_name)
    if caption == soundscript_name and not cc_only:
        caption = ''  # Redundant, choreo assumes the sound name.

    print(f'Writing {scene_name}, caption={caption!r}...')
    scene = choreo.Scene(actors=[
        choreo.Actor(
            name=actor_name,
            channels=[choreo.Channel("audio", events=[
                choreo.SpeakEvent(
                    name=soundscript_name,
                    start_time=0.0,
                    end_time=duration,
                    flags=choreo.EventFlags.FixedLength | choreo.EventFlags.Active,
                    parameters=('Default.Null' if cc_only else soundscript_name, '', ''),
                    caption_type=choreo.CaptionType.Master,
                    cc_token=caption,
                    ramp=choreo.Curve(),
                )
            ])],
        )
    ])
    buf = io.StringIO()
    await trio.to_thread.run_sync(scene.export_text, buf)
    # XT = Error if already present (manual?)
    try:
        async with await scene_name.open('xt') as f:
            await f.write(COMMENT)
            await f.write(buf.getvalue())
    except FileExistsError:
        raise
    except Exception:
        # If something happens during file saving, remove the malformed file.
        pathlib.Path(scene_name).unlink(missing_ok=True)
        raise
    else:
        SCENES.append(choreo.Entry.from_scene(
            scene_name.relative_to(settings.game_dir).as_posix(),
            scene,
        ))


async def check_existing(settings: Settings, filename: trio.Path) -> None:
    """Check existing choreo files to remove auto-generated ones."""
    async with await filename.open('r') as f:
        first_line = await f.readline()
    if 'auto-generated' in first_line:
        await filename.unlink()
    else:
        print(f'"{filename}" is manually authored.')
        data = await filename.read_text()
        scene = await trio.to_thread.run_sync(
            choreo.Scene.parse_text, Tokenizer(data, filename)
        )
        for sound in scene.used_sounds():
            MANUAL_SOUNDSCRIPTS.add(sound.casefold())
        SCENES.append(choreo.Entry.from_scene(
            filename.relative_to(settings.game_dir).as_posix(),
            scene,
        ))


async def make_soundscript(settings: Settings, actor: str) -> None:
    """Make a soundscript for this actor."""
    filename = settings.game_dir / 'scripts' / f'npc_sounds_auto_{actor}.txt'
    bufs = []
    sounds = sorted(CHOREO_SOUNDS[actor].values(), key=lambda s: s.name)
    if not sounds:
        return
    async with trio.open_nursery() as nursery:
        for sound in sounds:
            bufs.append(io.StringIO())
            nursery.start_soon(trio.to_thread.run_sync, sound.export, bufs[-1])

    print(f'Writing {filename}...')
    async with await filename.open('w') as f:
        for buf in bufs:
            await f.write(buf.getvalue())
            await f.write('\n')


async def read_settings(path: trio.Path) -> Settings:
    """Read the settings."""
    path = await path.resolve()
    print("Config file path: ", path)
    with open(path, encoding='utf8') as f:
        conf = Keyvalues.parse(f)
    wpm = conf.float('wpm', 100.0)
    game_dir = await trio.Path(path, '..', conf['gamedir']).resolve()

    return Settings(
        game_dir=game_dir,
        actor_names=Overridable.parse(
            conf, str,
            'actornames', 'actoroverrides',
        ),
        mixgroups=Overridable.parse(
            conf, str,
            'mixgroups', 'mixgroupoverrides',
        ),
        soundlevels=Overridable.parse(
            conf, lambda text: sndscript.split_float(
                text,
                sndscript.SOUND_LEVELS.__getitem__,
                sndscript.Level.SNDLVL_TALKING
            ),
            'soundlevels', 'soundleveloverrides',
        ),
        use_operator_stacks=conf.bool('use_operator_stacks', True),
        seconds_per_word=60.0 / wpm,
        scene_imports={
            game_dir / image.real_name: None
            if not image.has_children() and image.value == '*'
            else image.as_array()
            for image in conf.find_children('image_imports')
        },
        caption_override={
            kv.real_name: kv.value
            for kv in conf.find_children("captionoverrides")
        }
    )


async def check_captions(settings: Settings) -> None:
    """Parse the captions file, and build scenes for captions without sounds."""
    subtitles_file = settings.game_dir / 'resource/subtitles_english.txt'
    async with await subtitles_file.open('r', encoding='utf-16-le') as f:
        kv = Keyvalues.parse(await f.readlines(), 'resource/subtitles_english.txt')

    async with trio.open_nursery() as nursery:
        for tok in kv.find_children('lang', 'tokens'):
            nursery.start_soon(scene_from_subtitle, settings, tok.real_name, tok.value)


async def merge_scenes_image(image_path: trio.Path, scenes: list[str] | None) -> None:
    """Merge a scenes.image file into our choreo scenes."""
    data = await image_path.read_bytes()
    image = await trio.to_thread.run_sync(choreo.parse_scenes_image, io.BytesIO(data))
    if scenes is None:
        SCENES.extend(image.values())
        return
    for scene in scenes:
        try:
            entry = image[choreo.checksum_filename(scene)]
        except KeyError:
            raise ValueError(f'Could not import scene "{scene}"!') from None
        entry.filename = scene
        SCENES.append(entry)


class Args(Protocol):
    """Parsed arguments."""
    config: str
    subtitles: str | None
    func: Callable[[Settings, 'Args'], Awaitable[None]]


async def main(argv: list[str]) -> None:
    """Search for files."""
    parser = argparse.ArgumentParser(
        description="Generates choreo scenes and soundscripts from subtitle files."
    )
    parser.add_argument(
        "config",
        nargs='?',
        default="../gen_choreo.vdf",
        help="The location of the config file.",
    )
    subparsers = parser.add_subparsers(title="action", required=True)

    parser_gen = subparsers.add_parser('generate')
    parser_gen.set_defaults(func=rebuild_scenes)

    parser_vscript = subparsers.add_parser('vscript')
    parser_vscript.set_defaults(func=make_vscript)
    parser_vscript.add_argument(
        'subtitles',
        nargs='?',
        default=None,
        help='Subtitles file to read. If unset, read/write from clipboard.'
    )

    args: Args = parser.parse_args(argv)

    settings = await read_settings(trio.Path(args.config))
    print(f'Game folder: {settings.game_dir}')

    await args.func(settings, args)


async def rebuild_scenes(settings: Settings, args: Args) -> None:
    """Rebuild all scenes."""
    print('Removing existing auto scenes...')
    async with trio.open_nursery() as nursery:
        for scene in await (settings.game_dir / 'scenes').rglob('*.vcd'):
            nursery.start_soon(check_existing, settings, scene)

    if settings.scene_imports:
        print('Importing scenes.image scenes...')
        async with trio.open_nursery() as nursery:
            for image, scenes in settings.scene_imports.items():
                nursery.start_soon(merge_scenes_image, image, scenes)

    print('Manual soundscripts', MANUAL_SOUNDSCRIPTS)
    print('Generating scenes from sounds...')

    sound_folder = await (settings.game_dir / 'sound/npc').resolve()
    async with trio.open_nursery() as nursery:
        for sound in await sound_folder.rglob("*.wav"):
            nursery.start_soon(scene_from_sound, settings, sound_folder, sound)

    print('Generating scenes from captions...')
    await check_captions(settings)

    image_buf = io.BytesIO()
    scene_done = trio.Event()

    async def build_image() -> None:
        print(f'Building scenes.image ({len(SCENES)} scenes)...', flush=True)
        await trio.to_thread.run_sync(choreo.save_scenes_image_sync, image_buf, SCENES)
        scene_done.set()

    print('Writing soundscripts...')
    async with trio.open_nursery() as nursery:
        nursery.start_soon(build_image)  # Can run while making soundscripts.
        for actor in CHOREO_SOUNDS:
            nursery.start_soon(make_soundscript, settings, actor)

        await scene_done.wait()
        print('Writing scenes.image...')
        await (settings.game_dir / 'scenes/scenes.image').write_bytes(image_buf.getvalue())


async def make_vscript(settings: Settings, args: Args) -> None:
    """Generate a VScript block to play the subtitles passed in."""
    sub_fname = args.subtitles
    if sub_fname:
        use_clipboard = False
        async with await trio.open_file(sub_fname, 'r') as f:
            sub_data = await f.read()
    else:
        sub_fname = '<clipboard>'
        use_clipboard = True
        sub_data = pyperclip.paste()
    subtitles = Keyvalues.parse(sub_data, sub_fname)
    first = subtitles[0].real_name
    try:
        [_, *segments, last] = first.split('.')
    except ValueError:
        raise ValueError(f'Invalid subtitle key: {first}') from None
    num_match = re.search(r'[0-9]+$', last)
    if num_match is None:
        num_size = 1  # Don't pad.
    else:
        num_size = num_match.end() - num_match.start()
        last = last[:-num_size]  # Strip them.
    new_names = []
    segs = '.' + '.'.join(segments) if segments else ''
    for i, sub in enumerate(subtitles, 1):
        char = sub.real_name.split('.', 1)[0]
        new_names.append((char, f'{char}{segs}.{last}{i:0{num_size}}'))

    if any(sub.real_name != name for (c, name), sub in zip(new_names, subtitles, strict=True)):
        print('Renumbering: ')
        for (char, name), sub in zip(new_names, subtitles, strict=True):
            print(f'{sub.real_name} -> {name}')
            sub.name = name

    out = io.StringIO()
    subtitles.serialise(out)
    out.write('\n\n')
    out.write(f'SceneTable["{last}"] <- [\n\t')
    for (char, name) in new_names:
        choreo = pathlib.PurePath('scenes', 'npc', *name.split('.')).as_posix()
        out.write(f'{{\n\t\tvcd=CreateSceneEntity("{choreo}.vcd"),\n\t\tchar="{char}",\n\t}}, ')
    out.write('\n]')
    if use_clipboard:
        pyperclip.copy(out.getvalue())
    else:
        sys.stdout.write(out.getvalue())

if __name__ == '__main__':
    trio.run(main, sys.argv[1:])
