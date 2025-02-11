"""Generate VCD choreo scripts for each character sound."""
from collections import defaultdict
from collections.abc import Mapping
import argparse
import io
import pathlib
import re
import sys
import traceback

from srctools import Keyvalues, choreo, sndscript
from srctools.tokenizer import Tokenizer
import attrs
import pyglet.media
import trio


COMMENT = '// This choreo scene was auto-generated. Remove this comment if modifying this file.\n'

FILES = []

CHOREO_SOUNDS: dict[str, dict[str, sndscript.Sound]] = defaultdict(dict)
CHOREO_TO_SOUNDSCRIPT = str.maketrans({
    '/': '.',
    '\\': '.',
    '-': '_',
})
SCENES: list[choreo.Entry] = []
RE_CAPTION_CMD = re.compile('<[^>]+>')
RE_WORDS = re.compile(r'\S+')
MANUAL_SOUNDSCRIPTS: set[str] = set()


@attrs.define(kw_only=True)
class Settings:
    """General configuration."""
    game_dir: trio.Path
    # folder=character -> actor entity name, or "" to skip.
    char_to_actor: Mapping[str, str]
    # Soundscript -> actor entity name, overrides folder.
    actor_overrides: Mapping[str, str]
    # For subtitle -> choreo, the WPM to use.
    seconds_per_word: float
    # Character to the mixgroup to use.
    char_to_mixgroup: Mapping[str, str]
    # Lists of scenes.image files to merge into ours, with the filenames to add.
    scene_imports: Mapping[trio.Path, list[str]]
    # Whether stacks are allowed.
    use_operator_stacks: bool

    def get_actor(self, soundscript: str, character: str) -> str:
        """Get the actor entity for this soundscript."""
        try:
            return self.actor_overrides[soundscript]
        except KeyError:
            try:
                return self.char_to_actor[character]
            except KeyError:
                raise ValueError(f'Unknown character "{character}" for soundscript "{soundscript}"!')


async def scene_from_sound(settings: Settings, root: trio.Path, filename: trio.Path) -> None:
    """Build a soundscript and scene for a sound."""
    try:
        duration = (await trio.to_thread.run_sync(pyglet.media.load, str(filename))).duration
    except pyglet.media.DecodeException as exc:
        traceback.print_exception(type(exc), exc, None)
        return

    relative = filename.relative_to(root)

    scene_name = await trio.Path(settings.game_dir, 'scenes', 'npc', relative).with_suffix('.vcd').resolve()
    soundscript_name = str(relative.with_suffix('')).translate(CHOREO_TO_SOUNDSCRIPT).lower()

    if soundscript_name.casefold() in MANUAL_SOUNDSCRIPTS:
        return

    character = relative.parts[0]

    if settings.get_actor(soundscript_name, character) == "":
        return

    mixgroup = settings.char_to_mixgroup.get(character, f'{character}VO')

    # print(filename, '->', soundscript_name, scene_name, duration)
    CHOREO_SOUNDS[character][soundscript_name] = snd = sndscript.Sound(
        name=soundscript_name,
        sounds=[str(trio.Path('npc', relative)).replace('\\', '/')],
        channel=sndscript.Channel.VOICE,
        level=sndscript.Level.SNDLVL_NONE,
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

    if settings.get_actor(soundscript, character) == "":
        return

    CHOREO_SOUNDS[character][soundscript] = snd = sndscript.Sound(
        name=soundscript,
        sounds=["common/null.wav"],
        channel=sndscript.Channel.VOICE,
        level=sndscript.Level.SNDLVL_NONE,
    )
    if settings.use_operator_stacks:
        snd.stack_start = Keyvalues("start_stack", [
            Keyvalues("import_stack", "P2_null_start"),
        ])

    stripped = RE_CAPTION_CMD.sub('', caption)

    word_count = sum(1 for _ in RE_WORDS.finditer(stripped))
    duration = settings.seconds_per_word * word_count
    # print(f'Caption: {soundscript} = {word_count} words = {duration}')
    scene_name = trio.Path(settings.game_dir, 'scenes', 'npc', *soundscript.split('.')).with_suffix('.vcd')
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
    actor_name = settings.get_actor(soundscript_name, character)
    assert actor_name != "", character

    await scene_name.parent.mkdir(parents=True, exist_ok=True)

    print(f'Writing {scene_name}, cc_only={cc_only}...')
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
                    cc_token=soundscript_name if cc_only else '',
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


async def check_existing(filename: trio.Path) -> None:
    """Check existing choreo files to remove auto-generated ones."""
    async with await filename.open('r') as f:
        first_line = await f.readline()
    if 'auto-generated' in first_line:
        await filename.unlink()
    else:
        print(f'"{filename}" is manually authored.')
        data = await filename.read_text()
        scene = await trio.to_thread.run_sync(choreo.Scene.parse_text, Tokenizer(data, filename))
        for sound in scene.used_sounds():
            MANUAL_SOUNDSCRIPTS.add(sound.casefold())


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
    print(f"Config file path: ", path)
    with open(path) as f:
        conf = Keyvalues.parse(f)
    wpm = conf.float('wpm', 100.0)
    game_dir = await trio.Path(path, '..', conf['gamedir']).resolve()

    return Settings(
        game_dir=game_dir,
        char_to_actor={
            kv.real_name: kv.value
            for kv in conf.find_children('actornames')
        },
        use_operator_stacks=conf.bool('use_operator_stacks', True),
        actor_overrides={
            # Flip, so we can group similar overrides together and it'll look nice.
            kv.value: kv.real_name
            for kv in conf.find_children('actoroverrides')
        },
        char_to_mixgroup={
            kv.real_name: kv.value
            for kv in conf.find_children('mixgroups')
        },
        seconds_per_word=60.0 / wpm,
        scene_imports={
            game_dir / image.real_name: image.as_array()
            for image in conf.find_children('image_imports')
        },
    )


async def check_captions(settings: Settings) -> None:
    """Parse the captions file, and build scenes for captions without sounds."""
    subtitles_file = settings.game_dir / 'resource/subtitles_english.txt'
    async with await subtitles_file.open('r', encoding='utf-16-le') as f:
        kv = Keyvalues.parse(await f.readlines(), 'resource/subtitles_english.txt')

    async with trio.open_nursery() as nursery:
        for tok in kv.find_children('lang', 'tokens'):
            nursery.start_soon(scene_from_subtitle, settings, tok.real_name, tok.value)


async def merge_scenes_image(image_path: trio.Path, scenes: list[str]) -> None:
    """Merge a scenes.image file into our choreo scenes."""
    data = await image_path.read_bytes()
    image = await trio.to_thread.run_sync(choreo.parse_scenes_image, io.BytesIO(data))
    for scene in scenes:
        try:
            entry = image[choreo.checksum_filename(scene)]
        except KeyError:
            raise ValueError(f'Could not import scene "{scene}"!') from None
        entry.filename = scene
        SCENES.append(entry)


async def main(argv: list[str]) -> None:
    """Search for files."""
    parser = argparse.ArgumentParser(
        description="Generates choreo scenes and soundscripts from subtitle files."
    )
    parser.add_argument(
        "config",
        default="../gen_choreo.vdf",
        help="The location of the config file.",
    )

    args = parser.parse_args(argv)

    settings = await read_settings(trio.Path(args.config))
    print(f'Game folder: {settings.game_dir}')

    print('Removing existing auto scenes...')
    async with trio.open_nursery() as nursery:
        for scene in await (settings.game_dir / 'scenes').rglob('*.vcd'):
            nursery.start_soon(check_existing, scene)

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

if __name__ == '__main__':
    trio.run(main, sys.argv[1:])
