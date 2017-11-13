"""Handles the list of files which are desired to be packed into the BSP."""
from collections import deque

from enum import Enum
from zipfile import ZipFile
import os.path

from srctools.bsp import BSP
from srctools.filesys import FileSystem, VPKFileSystem, FileSystemChain
from srctools.vmt import Material, VarType

from typing import Container

FILES = {}  # List of all the files we are packing.


class FileType(Enum):
    """Types of files we might pack."""
    GENERIC = 0  # Other file types.
    SOUNDSCRIPT = 1  # Should be added to the manifest
    
    # Assume these files are present even
    # if we can't find them. (rt_ textures for example.)
    # also don't bother looking for dependencies.
    WHITELIST = 2
    
    # This file might be present - if it is pack it.
    # If not it's not an error.
    OPTIONAL = 3
    
    PARTICLE = 'pcf'  # Should be added to the manifest
    
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


class PackFile:
    """Represents a single file we are packing.
    
    data is raw data to pack directly, instead of from the filesystem.
    """
    __slots__ = ['type', 'filename', 'data', 'virtual']
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


def pack_file(
    filename: str,
    data_type: FileType=FileType.GENERIC, 
    data: bytes=None,
):
    """Queue the given file to be packed. 
    If data is set, this file will use the given data.
    This returns the PackFile definition created.
    """

    path = unify_path(filename)

    try:
        file = FILES[path]
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
    
    FILES[path] = file = PackFile(
        data_type,
        filename,
        data,
    )
    return file


def pack_from_bsp(bsp: BSP):
    """Pack files found in BSP data (excluding entities)."""
    for static_prop in bsp.static_props_list():
        pack_file(static_prop, FileType.MODEL)

    for mat in bsp.read_texture_names():
        pack_file('materials/{}.vmt'.format(mat.lower()), FileType.MATERIAL)


def pack_into_zip(
    filesys: FileSystemChain, 
    zip_file: ZipFile, 
    block_filesys: Container[FileSystem]=(),
    ignore_vpk=True,
):
    """Pack all our files into the given zipfile.
    
    The filesys is used to find files to pack.
    If set, limit_filesys will disallow packing from the listed filesystems.
    If ignore_vpk is True, files in VPK won't be packed.
    """
    with filesys:
        for file in FILES.values():
            if file.virtual:
                # Always pack.
                zip_file.writestr(file.filename, file.data)
                continue

            try:
                sys_file = filesys[file.filename]
            except FileNotFoundError:
                if file.type is not FileType.OPTIONAL:
                    print('WARNING: "{}" not packed!'.format(file.filename))
                return

            sys = filesys.get_system(sys_file)

            if ignore_vpk and isinstance(sys, VPKFileSystem):
                continue
            elif sys in block_filesys:
                continue

            with sys_file.open_bin() as f:
                zip_file.writestr(file.filename, f.read())


def eval_dependencies(filesys: FileSystem):
    """Add files to the list which need to also be packed.
    
    For several files this has the side effect of reading in their data, 
    which will be saved.
    """
    # Run through the deque from start to end, adding dependencies as we go.
    done = set()
    todo = deque(FILES.values())
    with filesys:
        while todo:
            file = todo.popleft()
            done.add(file)

            if file.type is FileType.MATERIAL:
                deps = get_material_files(filesys, file)
            elif file.type is FileType.MODEL:
                deps = get_model_files(filesys, file)
            elif file.type is FileType.TEXTURE:
                # Try packing the '.hdr.vtf' file as well if present.
                deps = [pack_file(file.filename[:-3] + 'hdr.vtf', FileType.OPTIONAL)]
            else:
                deps = ()

            for dep in deps:
                if dep not in done:
                    done.add(dep)
                    todo.append(dep)


def get_model_files(filesys: FileSystem, file: PackFile):
    """Find any needed files for a model."""
    filename, ext = os.path.splitext(file.filename)
    yield pack_file(filename + '.vvd')  # Must be present.

    # Some of these are optional.
    for ext in ['.phy', '.vtx', '.sw.vtx', '.dx80.vtx', '.dx90.vtx']:
        component = filename + ext
        if component in filesys:
            yield pack_file(component)


def get_material_files(filesys: FileSystem, file: PackFile):
    """Find any needed files for a material."""
    parents = []
    try:
        with filesys, filesys.open_str(file.filename) as f:
            mat = Material.parse(f, file.filename)
    except FileNotFoundError:
        print('WARNING: File "{}" does not exist!'.format(file.filename))
        return

    # For 'patch' shaders, apply the originals.
    mat = mat.apply_patches(filesys, parent_func=parents.append)

    for vmt in parents:
        yield pack_file(vmt, FileType.MATERIAL)

    for param_name, param_type, param_value in mat:
        param_value = param_value.casefold()
        if param_type is VarType.TEXTURE:
            # Skip over reference to cubemaps, or realtime buffers.
            if param_value == 'env_cubemap' or param_value.startswith('_rt_'):
                continue
            yield pack_file(
                'materials/' + param_value + '.vtf',
                FileType.TEXTURE,
            )
