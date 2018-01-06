"""Handles the list of files which are desired to be packed into the BSP."""
from typing import Container, Dict
from enum import Enum
from zipfile import ZipFile
import os.path

from srctools import VMF
from srctools.fgd import FGD, ValueTypes as KVTypes, KeyValues
from srctools.bsp import BSP
from srctools.filesys import FileSystem, VPKFileSystem, FileSystemChain
from srctools.mdl import Model
from srctools.vmt import Material, VarType
from seecompiler.logger import get_logger

LOGGER = get_logger(__name__)


class FileType(Enum):
    """Types of files we might pack."""
    GENERIC = 0  # Other file types.
    SOUNDSCRIPT = 1  # Should be added to the manifest

    GAME_SOUND = 2  # 'world.blah' sound - lookup the soundscript, and raw files.
    
    # Assume these files are present even
    # if we can't find them. (rt_ textures for example.)
    # also don't bother looking for dependencies.
    WHITELIST = 3
    
    # This file might be present - if it is pack it.
    # If not it's not an error.
    OPTIONAL = 4
    
    PARTICLE_FILE = 'pcf'  # Should be added to the manifest
    
    VSCRIPT_SQUIRREL = 'nut'
    
    # Implies packing referenced materials and textures.
    MATERIAL = 'vmt'
    
    TEXTURE = 'vtf'  # May want .hdr.vtf too.
    
    # Requires lookup of vtx, vvd, phy files too - in the model data.
    # also any skins used.
    MODEL = 'mdl'


EXT_TYPE = {
    '.' + filetype.value: filetype
    for filetype in FileType
    if isinstance(filetype.value, str)
}

# noinspection PyProtectedMember
from seecompiler._class_resources import CLASS_RESOURCES


class PackFile:
    """Represents a single file we are packing.
    
    data is raw data to pack directly, instead of from the filesystem.
    """
    __slots__ = ['type', 'filename', 'data', 'virtual', '_analysed']
    def __init__(
        self, 
        type: FileType,
        filename: str, 
        data: bytes=None,
    ):
        self.type = type
        self.filename = filename
        self.data = data
        self.virtual = data is not None
        # If we've checked for dependencies
        self._analysed = False

    def __repr__(self):
        text = '<{}{} Packfile "{}"'.format(
            'virtual ' if self.virtual else '',
            self.type.name,
            self.filename,
        )
        if self.data is not None:
            text += ' with {} bytes data>'.format(len(self.data))
        else:
            text += '>'
        return text

    def get_data(self, filesys: FileSystem) -> bytes:
        """Read this file in. 
        
        If we have associated data, that is returned.
        Otherwise the filesystem is consulted for this file,
        and that is returned as well as cached.
        """
        if self.data is None:
            with filesys, filesys.open_bin(self.filename) as f:
                self.data = f.read()
        return self.data


def unify_path(path: str):
    """Convert paths to a unique form."""
    path = os.path.normpath(path).casefold().replace('\\', '/')
    if '../' in path:
        raise ValueError('Path tried to escape root!')
    return path.lstrip('/')


class PackList:
    """Represents a list of resources for a map."""
    def __init__(self, fsys: FileSystemChain):
        self._files = {}  # type: Dict[str, PackFile]
        self.fsys = fsys
    def __getitem__(self, path: str):
        """Look up a packfile by filename."""
        return self._files[unify_path(path)]

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        return iter(self._files.values())

    def __contains__(self, path):
        return unify_path(path) in self._files

    def pack_file(
        self,
        filename: str,
        data_type: FileType=FileType.GENERIC,
        data: bytes=None,
    ):
        """Queue the given file to be packed.
        If data is set, this file will use the given data.
        """
        filename = os.fspath(filename)

        if '\t' in filename:
            raise ValueError

        if data_type is FileType.MATERIAL or filename.endswith('.vmt'):
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith(('.vmt', '.spr')):
                filename = filename + '.vmt'
        elif data_type is FileType.TEXTURE or filename.endswith('.vtf'):
            if not filename.startswith('materials/'):
                filename = 'materials/' + filename
            if not filename.endswith('.vtf'):
                filename = filename + '.vtf'

        path = unify_path(filename)

        try:
            file = self._files[path]
        except KeyError:
            pass
        else:
            # It's already here, is that OK?

            # Allow overriding data on disk with ours..
            if file.data is None:
                if data is not None:
                    file.data = data
                    file.virtual = True
                # else: no data on either
            elif data == file.data:
                pass  # Overrode with the same data, that's fine
            elif file.data:
                raise ValueError('"{}": two different data streams!'.format(filename))
            # else: we have an override, but asked to just pack.

            if file.type is data_type:
                # Same, no problems - just packing on top.
                return file

            if file.type is FileType.GENERIC:
                file.type = data_type  # This is fine, we now know it has behaviour...
            elif data_type is FileType.GENERIC:
                return file  # If we know it has behaviour, that trumps generic.

            if data_type is FileType.WHITELIST:
                file.type = data_type  # Blindly believe this.

            raise ValueError('"{}": {} can\'t become a {}!'.format(
                filename,
                file.type.name,
                data_type.name,
            ))

        start, ext = os.path.splitext(path)

        # Try to promote generic to other types if known.
        if data_type is FileType.GENERIC:
            try:
                data_type = EXT_TYPE[ext]
            except KeyError:
                pass
        elif data_type is FileType.SOUNDSCRIPT:
            if ext != '.txt':
                raise ValueError('"{}" cannot be a soundscript!'.format(filename))

        self._files[path] = file = PackFile(
            data_type,
            filename,
            data,
        )
        return file

    def pack_from_bsp(self, bsp: BSP):
        """Pack files found in BSP data (excluding entities)."""
        for static_prop in bsp.static_prop_models():
            self.pack_file(static_prop, FileType.MODEL)

        for mat in bsp.read_texture_names():
            self.pack_file('materials/{}.vmt'.format(mat.lower()), FileType.MATERIAL)

    def pack_fgd(self, vmf: VMF, fgd: FGD):
        """Analyse the map to pack files. We use the FGD to easily handle this."""
        for ent in vmf.entities:
            classname = ent['classname']
            try:
                ent_class = fgd[classname]
            except KeyError:
                LOGGER.warning('Unknown class "{}"!', classname)
                continue
            for key in set(ent.keys) | set(ent_class.kv):
                # These are always present on entities, and we don't have to do
                # any packing for them.
                # Origin/angles might be set (brushes, instances) even for ents
                # that don't use them.
                if key in ('classname', 'hammerid', 'origin', 'angles', 'skin', 'pitch'):
                    continue
                elif key == 'model':
                    # Models are set on all brush entities, and are always either
                    # a '*37' brush ref, a model, or a sprite.
                    value = ent[key]
                    if value and value[:1] != '*':
                        self.pack_file(value)
                    continue
                try:
                    kv = ent_class.kv[key]  # type: KeyValues
                    val_type = kv.type
                    default = kv.default
                except KeyError:
                    LOGGER.warning('Unknown keyvalue "{}" for ent of type "{}"!',
                                   key, ent['classname'])
                    val_type = None  # Doesn't match any enum.
                    default = ''

                value = ent[key, default]

                # Ignore blank values, they're not useful.
                if not value:
                    continue

                if classname == 'env_projectedtexture' and key == 'texturename':
                    # Special case - this is a VTF, not a material.
                    self.pack_file(value, FileType.TEXTURE)
                    continue

                if val_type is KVTypes.STR_MATERIAL:
                    self.pack_file(value, FileType.MATERIAL)
                elif val_type is KVTypes.STR_MODEL:
                    self.pack_file(value, FileType.MODEL)
                elif val_type is KVTypes.STR_VSCRIPT:
                    for script in value.split():
                        self.pack_file('scripts/vscripts/' + script)
                elif val_type is KVTypes.STR_SPRITE:
                    self.pack_file('materials/sprites/' + value, FileType.MATERIAL)

        for classname in vmf.by_class.keys():
            try:
                res = CLASS_RESOURCES[classname]
            except KeyError:
                continue
            if callable(res):
                for ent in vmf.by_class[classname]:
                    for file in res(ent):
                        self.pack_file(file)
            else:
                for file in res:
                    self.pack_file(file)

    def pack_into_zip(
        self,
        zip_file: ZipFile,
        block_filesys: Container[FileSystem]=(),
        ignore_vpk=True,
    ):
        """Pack all our files into the given zipfile.

        The filesys is used to find files to pack.
        If set, limit_filesys will disallow packing from the listed filesystems.
        If ignore_vpk is True, files in VPK won't be packed.
        """
        existing_names = set(zip_file.namelist())

        with self.fsys:
            for file in self._files.values():
                if file.virtual:
                    # Always pack.
                    zip_file.writestr(file.filename, file.data)
                    continue

                if file.filename in existing_names:
                    # Already in the zip - cubemap patch files, or something
                    # else has already added it. Ignore.
                    continue

                try:
                    sys_file = self.fsys[file.filename]
                except FileNotFoundError:
                    if file.type is not FileType.OPTIONAL:
                        print('WARNING: "{}" not packed!'.format(file.filename))
                    continue

                sys = self.fsys.get_system(sys_file)

                if ignore_vpk and isinstance(sys, VPKFileSystem):
                    continue
                elif sys in block_filesys:
                    continue

                with sys_file.open_bin() as f:
                    zip_file.writestr(file.filename, f.read())

    def eval_dependencies(self):
        """Add files to the list which need to also be packed.

        This requires parsing through many files.
        """
        # Run though repeatedly, until all are analysed.
        todo = True
        with self.fsys:
            while todo:
                todo = False
                for file in list(self._files.values()):
                    if file._analysed:
                        continue
                    file._analysed = True

                    if file.type is FileType.MATERIAL:
                        if self._get_material_files(file):
                            todo = True
                    elif file.type is FileType.MODEL:
                        self._get_model_files(file)
                        todo = True
                    elif file.type is FileType.TEXTURE:
                        # Try packing the '.hdr.vtf' file as well if present.
                        todo = True
                        self.pack_file(
                            file.filename[:-3] + 'hdr.vtf',
                            FileType.OPTIONAL
                        )

    def _get_model_files(self, file: PackFile):
        """Find any needed files for a model."""
        filename, ext = os.path.splitext(file.filename)
        self.pack_file(filename + '.vvd')  # Must be present.

        # Some of these are optional.
        for ext in ['.phy', '.vtx', '.sw.vtx', '.dx80.vtx', '.dx90.vtx']:
            component = filename + ext
            if component in self.fsys:
                yield self.pack_file(component)

        try:
            mdl = Model(self.fsys, self.fsys[file.filename])
        except FileNotFoundError:
            LOGGER.warning('Can\'t find model "{}"!', file.filename)
            return

        for tex in mdl.iter_textures():
            self.pack_file(tex, FileType.MATERIAL)

        for file in mdl.included_models:
            self.pack_file(file.filename, FileType.MODEL)

        return True  # Have dependencies

    def _get_material_files(self, file: PackFile):
        """Find any needed files for a material."""

        parents = []
        try:
            with self.fsys, self.fsys.open_str(file.filename) as f:
                mat = Material.parse(f, file.filename)
        except FileNotFoundError:
            print('WARNING: File "{}" does not exist!'.format(file.filename))
            return

        # For 'patch' shaders, apply the originals.
        mat = mat.apply_patches(self.fsys, parent_func=parents.append)

        for vmt in parents:
            yield self.pack_file(vmt, FileType.MATERIAL)

        has_deps = bool(parents)

        for param_name, param_type, param_value in mat:
            param_value = param_value.casefold()
            if param_type is VarType.TEXTURE:
                # Skip over reference to cubemaps, or realtime buffers.
                if param_value == 'env_cubemap' or param_value.startswith('_rt_'):
                    continue
                self.pack_file(
                    'materials/' + param_value + '.vtf',
                    FileType.TEXTURE,
                )
                has_deps = True

        return has_deps
